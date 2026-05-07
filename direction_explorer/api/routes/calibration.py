"""POST /calibration/compute — computes a direction (or N directions for SOM)
at a given layer, persists it to disk, and updates the in-memory store."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from direction_explorer.api.deps import (
    get_ablation,
    get_calibration,
    get_ctx,
    get_logit_lens,
    get_registry,
    get_repo,
    get_settings_dep,
    get_store,
)
from direction_explorer.api.schemas import CalibrationRequest
from direction_explorer.config import Settings
from direction_explorer.core.logit_lens import LogitLens
from direction_explorer.core.model_context import ModelContext
from direction_explorer.extractors.registry import ExtractorRegistry
from direction_explorer.persistence.direction_store import (
    CalibrationSet,
    DirectionStore,
)
from direction_explorer.persistence.disk_repository import DiskRepository
from direction_explorer.persistence.layer_keys import computed_layer_sort_key


router = APIRouter()


@router.post("/calibration/compute")
def calibration_compute(
    req: CalibrationRequest,
    settings: Settings = Depends(get_settings_dep),
    ctx: ModelContext = Depends(get_ctx),
    store: DirectionStore = Depends(get_store),
    registry: ExtractorRegistry = Depends(get_registry),
    repo: DiskRepository = Depends(get_repo),
    calibration: CalibrationSet = Depends(get_calibration),
    lens: LogitLens = Depends(get_logit_lens),
):
    if not req.harmful_prompts or not req.harmless_prompts:
        raise HTTPException(status_code=400, detail="Empty calibration set")
    if len(req.harmful_prompts) < 4 or len(req.harmless_prompts) < 4:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need at least 4 prompts per side (got "
                f"{len(req.harmful_prompts)} / {len(req.harmless_prompts)})."
            ),
        )
    if not (0 <= req.layer < ctx.n_layers):
        raise HTTPException(
            status_code=400, detail=f"layer must be in [0, {ctx.n_layers - 1}]",
        )
    method = req.extraction_method
    if method not in registry.methods():
        raise HTTPException(
            status_code=400,
            detail=(
                f"unknown extraction_method '{method}' "
                f"(allowed: {', '.join(registry.methods())})"
            ),
        )

    set_id = calibration.replace_if_changed(req.harmful_prompts, req.harmless_prompts)

    extractor = registry.get(method, ctx)
    t0 = time.time()
    result = extractor.compute(
        harmful_prompts=req.harmful_prompts,
        harmless_prompts=req.harmless_prompts,
        layer=req.layer,
        som_grid_rows=req.som_grid_rows,
        som_grid_cols=req.som_grid_cols,
        som_iterations=req.som_iterations,
        som_learning_rate=req.som_learning_rate,
        som_sigma=req.som_sigma,
        som_seed=req.som_seed,
    )

    if method == "mean_diff":
        return _respond_mean_diff(
            result, ctx, store, repo, calibration, lens, settings, t0,
        )

    if method == "som_md":
        return _respond_som(
            result, req, ctx, store, repo, calibration, lens, settings, t0,
        )

    # Future extractors: a generic responder would go here.
    raise HTTPException(status_code=500, detail=f"no responder for method '{method}'")


def _respond_mean_diff(
    result, ctx: ModelContext, store: DirectionStore, repo: DiskRepository,
    calibration: CalibrationSet, lens: LogitLens, settings: Settings, t0: float,
):
    ed = result.directions[0]
    direction = ed.direction
    raw_norm = ed.raw_norm
    normalized_score = ed.normalized_score
    layer = result.layer
    set_id = calibration.id

    top_tokens, bottom_tokens = lens.project(direction, k=10)

    entry = {
        "direction": direction.detach().to("cpu"),
        "raw_norm": raw_norm,
        "normalized_score": normalized_score,
        "top_tokens": top_tokens,
        "bottom_tokens": bottom_tokens,
        "calibration_set_id": set_id,
        "model_name": settings.model_name,
        "n_layers": ctx.n_layers,
        "d_model": ctx.d_model,
        "extraction_method": "mean_diff",
        "display_label": f"L{layer}",
    }
    store.put(layer, entry)

    repo.persist(
        layer_key=layer,
        direction=direction,
        raw_norm=raw_norm,
        normalized_score=normalized_score,
        top_tokens=top_tokens,
        bottom_tokens=bottom_tokens,
        set_id=set_id,
        extraction_method="mean_diff",
    )

    return {
        "method": "mean_diff",
        "layer": layer,
        "raw_norm": round(raw_norm, 4),
        "normalized_score": round(normalized_score, 4),
        "direction_shape": list(direction.shape),
        "direction_dtype": str(direction.dtype),
        "top_tokens": [{"token": t["token"], "score": round(t["score"], 4)}
                       for t in top_tokens],
        "bottom_tokens": [{"token": t["token"], "score": round(t["score"], 4)}
                          for t in bottom_tokens],
        "calibration_set_id": set_id,
        "elapsed_s": round(time.time() - t0, 2),
    }


def _respond_som(
    result, req: CalibrationRequest, ctx: ModelContext,
    store: DirectionStore, repo: DiskRepository,
    calibration: CalibrationSet, lens: LogitLens, settings: Settings, t0: float,
):
    layer = result.layer
    set_id = calibration.id
    rows = result.summary["som_grid_rows"]
    cols = result.summary["som_grid_cols"]

    response_neurons: list[dict] = []
    canonical_replaced = False
    for ed in result.directions:
        i = ed.metadata["neuron_index"]
        layer_key = ed.layer_key
        d_vec = ed.direction
        raw_norm_i = ed.raw_norm
        normalized_score_i = ed.normalized_score
        cluster_share = ed.metadata["cluster_share"]
        rank = ed.metadata["rank_by_cluster_size"]
        tightness_clean = ed.metadata.get("cluster_tightness")

        top_tokens_i, bottom_tokens_i = lens.project(d_vec, k=10)

        entry = {
            "direction": d_vec.detach().to("cpu"),
            "raw_norm": raw_norm_i,
            "normalized_score": normalized_score_i,
            "top_tokens": top_tokens_i,
            "bottom_tokens": bottom_tokens_i,
            "calibration_set_id": set_id,
            "model_name": settings.model_name,
            "n_layers": ctx.n_layers,
            "d_model": ctx.d_model,
            "extraction_method": "som_md",
            "lattice_position": list(ed.metadata["lattice_position"]),
            "neuron_index": i,
            "cluster_size": int(ed.metadata["cluster_size"]),
            "cluster_share": cluster_share,
            "cluster_tightness": tightness_clean,
            "som_grid_rows": rows,
            "som_grid_cols": cols,
            "display_label": f"L{layer} (SOM n[{i // cols},{i % cols}])",
        }
        store.put(layer_key, entry)

        repo.persist(
            layer_key=layer_key,
            direction=d_vec,
            raw_norm=raw_norm_i,
            normalized_score=normalized_score_i,
            top_tokens=top_tokens_i,
            bottom_tokens=bottom_tokens_i,
            set_id=set_id,
            extraction_method="som_md",
            extra_meta={
                "lattice_position": list(ed.metadata["lattice_position"]),
                "neuron_index": i,
                "cluster_size": int(ed.metadata["cluster_size"]),
                "cluster_share": cluster_share,
                "cluster_tightness": tightness_clean,
                "som_grid_rows": rows,
                "som_grid_cols": cols,
            },
        )

        # If user asked replace_canonical and this is the top-cluster neuron,
        # also write a canonical entry at int key.
        if req.replace_canonical and rank == 0:
            canonical_replaced = True
            store.put(layer, {
                "direction": d_vec.detach().to("cpu"),
                "raw_norm": raw_norm_i,
                "normalized_score": normalized_score_i,
                "top_tokens": top_tokens_i,
                "bottom_tokens": bottom_tokens_i,
                "calibration_set_id": set_id,
                "model_name": settings.model_name,
                "n_layers": ctx.n_layers,
                "d_model": ctx.d_model,
                "extraction_method": "som_md",
                "display_label": f"L{layer} (SOM canonical)",
            })
            repo.persist(
                layer_key=layer,
                direction=d_vec,
                raw_norm=raw_norm_i,
                normalized_score=normalized_score_i,
                top_tokens=top_tokens_i,
                bottom_tokens=bottom_tokens_i,
                set_id=set_id,
                extraction_method="som_md",
            )

        response_neurons.append({
            "neuron_index": i,
            "lattice_position": list(ed.metadata["lattice_position"]),
            "layer_key": layer_key,
            "is_canonical": (req.replace_canonical and rank == 0),
            "rank_by_cluster_size": rank,
            "raw_norm": round(raw_norm_i, 4),
            "cluster_size": int(ed.metadata["cluster_size"]),
            "cluster_share": round(cluster_share, 4),
            "cluster_tightness": (round(float(tightness_clean), 4)
                                  if tightness_clean is not None else None),
            "top_tokens": [{"token": t["token"], "score": round(t["score"], 4)}
                           for t in top_tokens_i],
            "bottom_tokens": [{"token": t["token"], "score": round(t["score"], 4)}
                              for t in bottom_tokens_i],
        })

    return {
        "method": "som_md",
        "layer": layer,
        "n_neurons": result.summary["n_neurons"],
        "som_grid_rows": rows,
        "som_grid_cols": cols,
        "harmful_centroid_norm": round(result.summary["harmful_centroid_norm"], 4),
        "harmless_centroid_norm": round(result.summary["harmless_centroid_norm"], 4),
        "canonical_replaced": canonical_replaced,
        "neurons": response_neurons,
        "calibration_set_id": set_id,
        "elapsed_s": round(time.time() - t0, 2),
    }
