"""
Model Loader — loads an LLM either from HuggingFace Hub (transformers) or
from a local GGUF file (llama-cpp-python).

Both paths are wrapped in try/except ImportError with a clear install
instruction, since llama-cpp-python in particular requires build tools and
isn't in requirements.txt by default (commented out — see requirements.txt).
"""

from dataclasses import dataclass
from typing import Any, Optional

from llm.supported_models import resolve


@dataclass
class LoadedModel:
    backend: str              # 'huggingface' | 'gguf'
    model: Any
    tokenizer: Optional[Any] = None
    repo_id: str = ""


def load_from_hub(model_name: str, device_map: str = "auto",
                   load_in_4bit: bool = False) -> LoadedModel:
    """Load a model + tokenizer from HuggingFace Hub via transformers.

    load_in_4bit requires bitsandbytes + a CUDA GPU (QLoRA). Falls back to
    full precision with a warning if bitsandbytes isn't installed.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers is required for HuggingFace Hub models. "
            "Install with: pip install transformers accelerate"
        ) from e

    repo_id = resolve(model_name)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

    kwargs: dict = {"device_map": device_map, "trust_remote_code": True}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            print(
                "⚠️  bitsandbytes not installed — falling back to full precision. "
                "Install with: pip install bitsandbytes (Linux/CUDA only)"
            )

    model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)
    return LoadedModel(backend="huggingface", model=model, tokenizer=tokenizer, repo_id=repo_id)


def load_from_gguf(gguf_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0) -> LoadedModel:
    """Load a quantized GGUF model via llama-cpp-python.

    NOTE: llama-cpp-python is NOT in requirements.txt (build-tool dependency,
    commented out there). Install manually:
        pip install llama-cpp-python
    or with GPU offload:
        CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
    """
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            "llama-cpp-python is required for GGUF models but is not installed "
            "(it's commented out in requirements.txt because it needs build tools). "
            "Install with: pip install llama-cpp-python"
        ) from e

    model = Llama(model_path=gguf_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
    return LoadedModel(backend="gguf", model=model, tokenizer=None, repo_id=gguf_path)


def load(model_name: str, model_format: str = "huggingface", **kwargs) -> LoadedModel:
    if model_format == "huggingface":
        return load_from_hub(model_name, **kwargs)
    if model_format == "gguf":
        return load_from_gguf(model_name, **kwargs)
    raise ValueError(f"Unknown model_format: {model_format!r} (expected 'huggingface' or 'gguf')")
