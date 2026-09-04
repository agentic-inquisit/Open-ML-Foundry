"""
LoRA/QLoRA Trainer — fine-tunes a loaded causal LM with a PEFT LoRA adapter,
reporting progress through a callback (used by session_api.py to write
chat-style session events after every epoch/step).

Dataset format: JSONL, one object per line, either:
    {"prompt": "...", "completion": "..."}
or:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

from __future__ import annotations  # PEP 585 generics (list[X]) on Python 3.8

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from llm.model_loader import LoadedModel


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[list[str]] = None  # None = library default for the model arch
    epochs: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 1
    max_seq_length: int = 512
    output_dir: str = "training_outputs/lora"


ProgressCallback = Callable[[dict], None]  # receives {"epoch": int, "loss": float, ...}


def _read_jsonl_as_text(dataset_path: str, tokenizer) -> list[str]:
    texts = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "messages" in row:
                text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
            elif "prompt" in row and "completion" in row:
                text = row["prompt"] + row["completion"]
            else:
                raise ValueError(
                    f"Unrecognized row shape in {dataset_path}: expected "
                    f"'messages' or 'prompt'+'completion' keys, got {list(row.keys())}"
                )
            texts.append(text)
    return texts


def train(loaded: LoadedModel, dataset_path: str, config: LoRAConfig,
          on_progress: Optional[ProgressCallback] = None) -> dict:
    """Run LoRA fine-tuning. Returns a metrics dict on success.

    Raises on missing deps (peft/datasets) or malformed dataset rather than
    silently no-op'ing — a session's train_failed event should carry a real
    error, not a fabricated "success".
    """
    if loaded.backend != "huggingface":
        raise ValueError(
            "LoRA training requires a HuggingFace-backend model "
            "(GGUF models are inference-only via llama-cpp-python)."
        )

    try:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            "peft and datasets are required for LoRA training. "
            "Install with: pip install peft datasets"
        ) from e

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    texts = _read_jsonl_as_text(dataset_path, loaded.tokenizer)
    if not texts:
        raise ValueError(f"Dataset {dataset_path} contained no usable rows")

    tokenizer = loaded.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config.max_seq_length,
                          padding="max_length")

    dataset = Dataset.from_dict({"text": texts}).map(tokenize, batched=True)

    peft_config = PeftLoraConfig(
        r=config.r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=config.target_modules, task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(loaded.model, peft_config)

    class ProgressReportingTrainer(Trainer):
        def log(self, logs: dict, *args, **kwargs) -> None:
            super().log(logs, *args, **kwargs)
            if on_progress and "loss" in logs:
                on_progress({
                    "epoch": logs.get("epoch"),
                    "loss": logs.get("loss"),
                    "learning_rate": logs.get("learning_rate"),
                    "step": logs.get("step"),
                })

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
    )

    trainer = ProgressReportingTrainer(
        model=peft_model, args=args, train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    result = trainer.train()

    adapter_path = str(Path(config.output_dir) / "adapter")
    peft_model.save_pretrained(adapter_path)

    return {
        "final_loss": result.training_loss,
        "adapter_path": adapter_path,
        "epochs": config.epochs,
        "num_samples": len(texts),
    }
