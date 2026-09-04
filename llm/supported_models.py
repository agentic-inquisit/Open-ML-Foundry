"""
Supported LLM Registry

Curated list of models shown in the session UI's model picker. Users are not
limited to this list — any HuggingFace Hub repo id or local GGUF path works
via `custom_model_name` in the session creation request.

Repo ids below follow each vendor's historical naming convention but are not
yet confirmed against HuggingFace Hub; `verified=False` tracks that per
entry until someone checks it (visit the URL, or run
`huggingface-cli repo info <repo_id>`). An unresolved id fails at download
time with a clear error, not silently.
"""

from __future__ import annotations  # PEP 585 generics (list[X]) on Python 3.8

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelEntry:
    display_name: str
    repo_id: str                  # HF Hub repo id — see `verified` before trusting
    family: str                  # 'qwen' | 'glm' | 'kimi' | 'minimax' | 'deepseek' | 'gemma'
    size_hint: str                # e.g. "8B", "flash", "unknown"
    supports_gguf: bool = True    # Whether GGUF quantized builds are expected to exist
    verified: bool = False        # Has anyone confirmed repo_id resolves on HF Hub?
    notes: str = ""


SUPPORTED_MODELS: list[ModelEntry] = [
    ModelEntry(
        display_name="Qwen3.8",
        repo_id="Qwen/Qwen3.8-Instruct",
        family="qwen",
        size_hint="unknown",
        notes="Follows Qwen's Qwen/Qwen{N}-Instruct naming convention.",
    ),
    ModelEntry(
        display_name="GLM-5.3-Flash",
        repo_id="THUDM/glm-5.3-flash",
        family="glm",
        size_hint="flash",
        notes="Follows Zhipu/THUDM's glm-{N} naming convention.",
    ),
    ModelEntry(
        display_name="Kimi K3",
        repo_id="moonshotai/Kimi-K3",
        family="kimi",
        size_hint="unknown",
        notes="Follows Moonshot AI's Kimi-K{N} naming convention.",
    ),
    ModelEntry(
        display_name="MiniMax-H3",
        repo_id="MiniMaxAI/MiniMax-H3",
        family="minimax",
        size_hint="unknown",
        notes="Follows MiniMax's MiniMax-{name} naming convention.",
    ),
    ModelEntry(
        display_name="DeepSeek-V4",
        repo_id="deepseek-ai/DeepSeek-V4",
        family="deepseek",
        size_hint="unknown",
        notes="Follows DeepSeek's DeepSeek-V{N} naming convention.",
    ),
    ModelEntry(
        display_name="Gemma 4",
        repo_id="google/gemma-4",
        family="gemma",
        size_hint="unknown",
        notes="Follows Google's gemma-{N} naming convention.",
    ),
]


def list_models() -> list[dict]:
    return [
        {
            "display_name": m.display_name, "repo_id": m.repo_id, "family": m.family,
            "size_hint": m.size_hint, "supports_gguf": m.supports_gguf,
            "verified": m.verified, "notes": m.notes,
        }
        for m in SUPPORTED_MODELS
    ]


def resolve(display_name_or_repo_id: str) -> str:
    """Map a display name to its repo_id; pass through anything else unchanged
    so custom/unlisted HF repo ids and local GGUF paths keep working."""
    for m in SUPPORTED_MODELS:
        if display_name_or_repo_id in (m.display_name, m.repo_id):
            return m.repo_id
    return display_name_or_repo_id
