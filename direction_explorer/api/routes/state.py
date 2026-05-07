"""GET /state — returns current model + cached directions snapshot.

Response shape is part of the public contract; do not change field
names / types / nesting without coordinating with the UI."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from direction_explorer.api.deps import (
    get_calibration,
    get_ctx,
    get_settings_dep,
    get_store,
)
from direction_explorer.config import Settings
from direction_explorer.core.model_context import ModelContext
from direction_explorer.persistence.direction_store import (
    CalibrationSet,
    DirectionStore,
)
from direction_explorer.persistence.layer_keys import (
    base_layer_int,
    direction_kind,
    layer_label,
)


router = APIRouter()


@router.get("/state")
def get_state(
    settings: Settings = Depends(get_settings_dep),
    ctx: ModelContext = Depends(get_ctx),
    store: DirectionStore = Depends(get_store),
    calibration: CalibrationSet = Depends(get_calibration),
):
    dirs = []
    valid_layer_keys = []
    for lid, info in store.items_sorted():
        saved_model = info.get("model_name")
        if saved_model is not None and saved_model != settings.model_name:
            print(
                f"[CALIB] Skipping cached direction {lid}: "
                f"model mismatch ({saved_model} != {settings.model_name})"
            )
            continue
        method = info.get("extraction_method", "mean_diff")
        try:
            layer_int = base_layer_int(lid)
        except Exception:
            layer_int = -1
        dirs.append({
            "layer": layer_int,
            "layer_key": str(lid),
            "label": layer_label(lid, info),
            "kind": direction_kind(lid, info),
            "extraction_method": method,
            "raw_norm": info["raw_norm"],
            "normalized_score": info["normalized_score"],
            "calibration_set_id": info["calibration_set_id"],
            "top_tokens": info.get("top_tokens", []),
            "bottom_tokens": info.get("bottom_tokens", []),
            "lattice_position": info.get("lattice_position"),
            "neuron_index": info.get("neuron_index"),
            "cluster_size": info.get("cluster_size"),
            "cluster_share": info.get("cluster_share"),
            "cluster_tightness": info.get("cluster_tightness"),
            "som_grid_rows": info.get("som_grid_rows"),
            "som_grid_cols": info.get("som_grid_cols"),
        })
        valid_layer_keys.append(str(lid))
    return {
        "n_layers": ctx.n_layers,
        "d_model": ctx.d_model,
        "model": settings.model_name,
        "architecture": "Mistral",
        "device": ctx.device,
        "tokenizer_vocab": int(ctx.lm_head.weight.shape[0]),
        "directions": dirs,
        "computed_layers": valid_layer_keys,
        "calibration_set_id": calibration.id,
    }
