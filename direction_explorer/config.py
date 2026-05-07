"""Centralized settings — replaces scattered os.environ lookups.

A frozen dataclass instead of pydantic-settings so we don't add a new dep.
Same env var names and semantics as the original monolithic file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch


@dataclass(frozen=True)
class Settings:
    model_name: str
    hf_token: Optional[str]
    port: int
    device: str
    load_in_8bit: bool
    dtype: torch.dtype
    default_max_new_tokens: int
    max_new_tokens_cap: int
    results_dir: Path
    top_k: int = 12

    @property
    def safe_model_slug(self) -> str:
        return self.model_name.replace("/", "_").replace(":", "_")


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    device = _detect_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    return Settings(
        model_name=os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct"),
        hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        port=int(os.environ.get("PORT", "8002")),
        device=device,
        load_in_8bit=os.environ.get("LOAD_IN_8BIT", "0") == "1",
        dtype=dtype,
        default_max_new_tokens=int(os.environ.get("DEFAULT_MAX_NEW_TOKENS", "128")),
        max_new_tokens_cap=int(os.environ.get("MAX_NEW_TOKENS_CAP", "512")),
        results_dir=results_dir,
    )
