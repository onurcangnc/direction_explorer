"""Last-token residual collection at a single decoder layer.

These helpers register a transient forward hook, run the model on each
prompt, capture the last-token hidden state at `layer_idx`, then unregister
the hook. fp32 on CPU for activation matrices, fp32 on device for the mean
accumulator."""

from __future__ import annotations

import torch

from direction_explorer.core.model_context import ModelContext
from direction_explorer.core.prompt_formatting import format_prompt


def compute_single_layer_mean(
    ctx: ModelContext, prompts: list[str], layer_idx: int,
) -> torch.Tensor:
    """Mean last-token residual at `layer_idx` over `prompts`. fp32 on device."""
    accum = torch.zeros(ctx.d_model, dtype=torch.float32, device=ctx.device)
    captured: dict = {}

    def cap_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["last"] = h[:, -1, :].detach().to(torch.float32)

    handle = ctx.layers[layer_idx].register_forward_hook(cap_hook)
    try:
        for p in prompts:
            inputs = format_prompt(ctx, p)
            with torch.no_grad():
                _ = ctx.model(**inputs)
            accum += captured["last"][0]
    finally:
        handle.remove()
    return accum / max(len(prompts), 1)


def collect_layer_activations(
    ctx: ModelContext, prompts: list[str], layer_idx: int,
) -> torch.Tensor:
    """Return [n_prompts, d_model] fp32 CPU tensor of per-prompt last-token
    residuals at `layer_idx`. Used by SOM / future SVD extractors."""
    n = len(prompts)
    out = torch.zeros((n, ctx.d_model), dtype=torch.float32)
    captured: dict = {}

    def cap_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["last"] = h[:, -1, :].detach().to(torch.float32).cpu()

    handle = ctx.layers[layer_idx].register_forward_hook(cap_hook)
    try:
        for i, p in enumerate(prompts):
            inputs = format_prompt(ctx, p)
            with torch.no_grad():
                _ = ctx.model(**inputs)
            out[i] = captured["last"][0]
    finally:
        handle.remove()
    return out
