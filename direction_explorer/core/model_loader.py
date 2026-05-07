"""Load a HuggingFace causal LM into a ModelContext.

Preserves the exact same code paths and log output as the pre-refactor
monolith (CUDA + 8-bit, CUDA fp16, MPS / CPU)."""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from direction_explorer.config import Settings
from direction_explorer.core.model_context import ModelContext


def load_model_context(settings: Settings) -> ModelContext:
    print(f"[LOAD] {settings.model_name} on {settings.device} ({settings.dtype})...")
    t0 = time.time()
    if settings.device == "cuda" and settings.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
            token=settings.hf_token,
        )
    elif settings.device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            dtype=torch.float16,
            device_map={"": 0},
            trust_remote_code=False,
            token=settings.hf_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            dtype=settings.dtype,
            trust_remote_code=False,
            token=settings.hf_token,
        ).to(settings.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_name, trust_remote_code=False, token=settings.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_layers = len(model.model.layers)
    d_model = int(model.config.hidden_size)
    print(
        f"[LOAD] Done in {time.time() - t0:.1f}s. "
        f"Layers: {n_layers}, d_model: {d_model}"
    )
    return ModelContext(
        model=model,
        tokenizer=tokenizer,
        device=settings.device,
        dtype=settings.dtype,
        d_model=d_model,
        n_layers=n_layers,
        model_name=settings.model_name,
    )
