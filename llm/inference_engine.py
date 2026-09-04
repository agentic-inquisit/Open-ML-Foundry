"""
Inference Engine — chat-style generation against a loaded model (HF or
GGUF), optionally with a LoRA adapter applied. Used by the session "test in
chat" flow: the user types a prompt, this returns the model's reply as one
more event in the session history.
"""

import time
from typing import Optional

from llm.model_loader import LoadedModel


def generate(loaded: LoadedModel, prompt: str, max_tokens: int = 512,
             temperature: float = 0.7, adapter_path: Optional[str] = None) -> dict:
    """Generate a completion. Returns {"output": str, "latency_ms": float}."""
    start = time.time()

    if loaded.backend == "huggingface":
        output = _generate_hf(loaded, prompt, max_tokens, temperature, adapter_path)
    elif loaded.backend == "gguf":
        output = _generate_gguf(loaded, prompt, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown backend: {loaded.backend!r}")

    latency_ms = (time.time() - start) * 1000
    return {"output": output, "latency_ms": round(latency_ms, 1)}


def _generate_hf(loaded: LoadedModel, prompt: str, max_tokens: int,
                  temperature: float, adapter_path: Optional[str]) -> str:
    import torch

    model = loaded.model
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(loaded.model, adapter_path)

    tokenizer = loaded.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=temperature,
            do_sample=temperature > 0, pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _generate_gguf(loaded: LoadedModel, prompt: str, max_tokens: int, temperature: float) -> str:
    result = loaded.model(prompt, max_tokens=max_tokens, temperature=temperature)
    return result["choices"][0]["text"]
