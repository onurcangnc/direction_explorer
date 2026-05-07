"""HF text generation + per-token projection capture."""

from __future__ import annotations

import torch

from direction_explorer.core.model_context import ModelContext


def hf_generate(
    ctx: ModelContext,
    formatted_text: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, torch.Tensor, int]:
    """Generate continuation. Returns (response_text, full_token_ids, prompt_len)."""
    inputs = ctx.tokenizer(formatted_text, return_tensors="pt").to(ctx.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = ctx.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0.01,
            top_p=0.9,
            pad_token_id=ctx.tokenizer.eos_token_id,
        )
    full_ids = out[0]
    new_ids = full_ids[prompt_len:]
    response_text = ctx.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return response_text, full_ids, prompt_len


def capture_token_projections(
    ctx: ModelContext,
    full_token_ids: torch.Tensor,
    prompt_len: int,
    layer_idx: int,
    direction: torch.Tensor,
    ablation_layer_hook=None,
) -> tuple[list[str], list[float]]:
    """Re-run forward on (prompt + completion), capture residual at
    `layer_idx` for every position, project onto `direction`. Returns
    (token_strs, projs) for the GENERATED portion only.

    If `ablation_layer_hook` is provided, it's registered on every decoder
    layer BEFORE the capture hook, so the capture sees post-ablation
    activations.
    """
    if full_token_ids.shape[0] <= prompt_len:
        return [], []

    handles = []
    if ablation_layer_hook is not None:
        for layer in ctx.layers:
            handles.append(layer.register_forward_hook(ablation_layer_hook))

    captured: dict = {}

    def cap_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["resid"] = h.detach().clone()

    cap_handle = ctx.layers[layer_idx].register_forward_hook(cap_hook)
    handles.append(cap_handle)

    try:
        with torch.no_grad():
            ctx.model(input_ids=full_token_ids.unsqueeze(0).to(ctx.device))
    finally:
        for h in handles:
            h.remove()

    resid = captured["resid"][0]  # [seq, d_model]
    d = direction.to(resid.dtype).to(resid.device)
    projs_all = (resid @ d).detach().float().cpu().tolist()
    gen_projs = [round(x, 4) for x in projs_all[prompt_len:]]

    gen_ids = full_token_ids[prompt_len:].tolist()
    gen_strs = []
    for tid in gen_ids:
        try:
            gen_strs.append(ctx.tokenizer.decode([int(tid)]))
        except Exception:
            gen_strs.append(f"<{int(tid)}>")
    return gen_strs, gen_projs
