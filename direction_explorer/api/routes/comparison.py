"""POST /comparison/analyze — pairwise cosine similarity + token overlap
across cached directions."""

from __future__ import annotations

import torch
from fastapi import APIRouter, Depends, HTTPException

from direction_explorer.api.deps import get_store
from direction_explorer.api.schemas import ComparisonRequest
from direction_explorer.persistence.direction_store import DirectionStore
from direction_explorer.persistence.layer_keys import (
    computed_layer_sort_key,
    layer_label,
    parse_layer_key,
)


router = APIRouter()


@router.post("/comparison/analyze")
def comparison_analyze(
    req: ComparisonRequest,
    store: DirectionStore = Depends(get_store),
):
    parsed = []
    seen = set()
    for raw in req.layers:
        k = parse_layer_key(raw)
        if k in seen:
            continue
        seen.add(k)
        parsed.append(k)
    layers = sorted(parsed, key=computed_layer_sort_key)

    missing = [l for l in layers if not store.exists(l)]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Directions not computed for layer-key(s) {missing}. Compute them first.",
        )
    if len(layers) < 2:
        raise HTTPException(status_code=400, detail="Pick at least 2 layers.")

    dirs = [store.get(l)["direction"] for l in layers]
    stacked = torch.stack(dirs).float()
    norms = stacked.norm(dim=1, keepdim=True) + 1e-9
    normed = stacked / norms
    cos = (normed @ normed.t()).cpu().tolist()
    cos_rounded = [[round(float(v), 4) for v in row] for row in cos]

    labels = [layer_label(l, store.get(l) or {}) for l in layers]
    norm_data = [
        {
            "layer_key": str(l),
            "label": layer_label(l, store.get(l) or {}),
            "raw_norm": round(store.get(l)["raw_norm"], 4),
            "normalized_score": round(store.get(l)["normalized_score"], 4),
        }
        for l in layers
    ]

    overlap = []
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            la, lb = layers[i], layers[j]
            top_a = {t["token"] for t in store.get(la)["top_tokens"]}
            top_b = {t["token"] for t in store.get(lb)["top_tokens"]}
            shared = sorted(top_a & top_b)
            overlap.append({
                "layer_a": str(la),
                "layer_b": str(lb),
                "label_a": layer_label(la, store.get(la) or {}),
                "label_b": layer_label(lb, store.get(lb) or {}),
                "shared_tokens": shared,
                "count": len(shared),
            })

    return {
        "layers": [str(l) for l in layers],
        "labels": labels,
        "cosine_matrix": cos_rounded,
        "norms": norm_data,
        "top_token_overlap": overlap,
    }
