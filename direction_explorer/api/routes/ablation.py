"""POST /ablation/generate — baseline + ablated generation pair."""

from __future__ import annotations

import time

import torch
from fastapi import APIRouter, Depends, HTTPException

from direction_explorer.ablation.service import AblationService
from direction_explorer.ablation.strategies import AblationStrategyFactory
from direction_explorer.api.deps import (
    get_ablation,
    get_ctx,
    get_settings_dep,
    get_store,
)
from direction_explorer.api.schemas import AblationRequest
from direction_explorer.config import Settings
from direction_explorer.core.model_context import ModelContext
from direction_explorer.core.prompt_formatting import format_chat_text
from direction_explorer.persistence.direction_store import DirectionStore
from direction_explorer.persistence.layer_keys import (
    base_layer_int,
    layer_label,
    parse_layer_key,
)


router = APIRouter()


def _resolve_ablation_layers(req: AblationRequest, store: DirectionStore) -> list:
    """Returns deduplicated, ordered list of layer-keys to co-ablate.

    Primary first (parsed via `parse_layer_key`, so SOM/SVD string keys are
    preserved as strings — only canonical mean_diff lands as int), then
    extras in user order. All keys must already exist in the store.
    """
    seen = set()
    ordered: list = []
    primary = parse_layer_key(req.direction_layer)
    for raw in [primary, *req.extra_direction_layers]:
        k = parse_layer_key(raw)
        if k in seen:
            continue
        seen.add(k)
        ordered.append(k)
    missing = [l for l in ordered if not store.exists(l)]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"No cached direction(s) for layer-key(s) {missing}. "
                   "Compute them on the Calibration tab first.",
        )
    return ordered


@router.post("/ablation/generate")
def ablation_generate(
    req: AblationRequest,
    settings: Settings = Depends(get_settings_dep),
    ctx: ModelContext = Depends(get_ctx),
    store: DirectionStore = Depends(get_store),
    ablation: AblationService = Depends(get_ablation),
):
    if req.mode not in AblationStrategyFactory.modes():
        raise HTTPException(status_code=400, detail=f"unknown mode {req.mode}")

    ablation_layers = _resolve_ablation_layers(req, store)
    primary_key = parse_layer_key(req.direction_layer)
    primary_layer_int = base_layer_int(primary_key)
    primary_direction = store.get(primary_key)["direction"].to(ctx.device)
    all_directions = [
        store.get(lid)["direction"].to(ctx.device) for lid in ablation_layers
    ]

    eff_max_new = min(max(int(req.max_new_tokens), 1), settings.max_new_tokens_cap)

    gen_t0 = time.time()
    formatted = format_chat_text(ctx.tokenizer, req.prompt)
    prompt_token_len = ctx.tokenizer(
        formatted, return_tensors="pt",
    )["input_ids"].shape[1]

    try:
        result = ablation.run(
            prompt=req.prompt,
            primary_direction=primary_direction,
            all_directions=all_directions,
            primary_layer_int=primary_layer_int,
            mode=req.mode,
            strength=req.strength,
            max_new_tokens=eff_max_new,
            temperature=req.temperature,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ablation generation failed: {e}")

    total_elapsed = round(time.time() - gen_t0, 2)
    vram_peak_str = ""
    if ctx.device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        vram_peak_str = f", VRAM peak={peak_gb:.2f}GB"
    layers_repr = ",".join(layer_label(l, store.get(l) or {}) for l in ablation_layers)
    print(
        f"[GEN] mode={req.mode} primary=L{primary_layer_int} "
        f"co-ablated=[{layers_repr}] (k={len(ablation_layers)}) "
        f"prompt_len={prompt_token_len} "
        f"max_new={eff_max_new} (req={req.max_new_tokens}) "
        f"elapsed={total_elapsed}s "
        f"(base={result.elapsed_baseline_s}s, abl={result.elapsed_ablated_s}s)"
        f"{vram_peak_str}"
    )

    return {
        "baseline_response": result.baseline_response,
        "ablated_response": result.ablated_response,
        "baseline_tokens": result.baseline_tokens,
        "ablated_tokens": result.ablated_tokens,
        "baseline_token_projections": result.baseline_token_projections,
        "ablated_token_projections": result.ablated_token_projections,
        "elapsed_baseline_s": result.elapsed_baseline_s,
        "elapsed_ablated_s": result.elapsed_ablated_s,
        "mode": req.mode,
        "direction_layer": primary_layer_int,
        "direction_layers": [str(l) for l in ablation_layers],
        "direction_labels": [
            layer_label(l, store.get(l) or {}) for l in ablation_layers
        ],
        "strength": req.strength,
    }
