"""FastAPI app factory — the only place that knows how to wire everything
together. Module-level singletons are forbidden; all shared state lives on
the AppState the factory creates."""

from __future__ import annotations

import json
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from direction_explorer.ablation.service import AblationService
from direction_explorer.ablation.weight_snapshot import WeightSnapshot
from direction_explorer.api.deps import AppState, set_state
from direction_explorer.api.routes import all_routers
from direction_explorer.config import Settings, get_settings
from direction_explorer.core.logit_lens import LogitLens
from direction_explorer.core.model_context import ModelContext
from direction_explorer.core.model_loader import load_model_context
from direction_explorer.extractors.registry import default_registry
from direction_explorer.persistence.direction_store import (
    CalibrationSet,
    InMemoryDirectionStore,
)
from direction_explorer.persistence.disk_repository import DiskRepository
from direction_explorer.persistence.layer_keys import computed_layer_sort_key
from direction_explorer.prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
from direction_explorer.ui import render_index


def _print_startup_banner(settings: Settings) -> None:
    print(f"[STARTUP] Loading model: {settings.model_name}")
    print(f"[STARTUP] Port: {settings.port}")
    print(f"[CONFIG] Model: {settings.model_name}")
    print(f"[CONFIG] Device: {settings.device}, DType: {settings.dtype}")
    print(f"[CONFIG] 8-bit quantization: {settings.load_in_8bit}")
    print(f"[CONFIG] HF_TOKEN: {'set' if settings.hf_token else 'not set'}")
    if settings.device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"[CONFIG] VRAM free/total: {free/1e9:.2f}GB / {total/1e9:.2f}GB")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    _print_startup_banner(settings)

    ctx = load_model_context(settings)
    if settings.device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"[LOAD] VRAM allocated after load: {allocated:.2f}GB")
    print(f"[LOAD] Done. n_layers={ctx.n_layers}, d_model={ctx.d_model}")

    snapshot = WeightSnapshot(ctx)

    store = InMemoryDirectionStore()
    calibration = CalibrationSet(
        harmful=DEFAULT_HARMFUL,
        harmless=DEFAULT_HARMLESS,
        set_id=0,
    )
    repo = DiskRepository(settings, ctx.n_layers, ctx.d_model)
    n_loaded = repo.load_into(store, calibration)
    if n_loaded:
        print(f"[PERSIST] Loaded {n_loaded} cached direction(s) from {settings.results_dir}")
        print(f"[PERSIST] Layers: {sorted(store.keys(), key=computed_layer_sort_key)}")
    else:
        print(f"[PERSIST] No cached directions found in {settings.results_dir}")

    print(f"Direction Explorer ready ({settings.model_name}, manual PyTorch hooks)")

    state = AppState(
        settings=settings,
        ctx=ctx,
        store=store,
        registry=default_registry(),
        ablation=AblationService(ctx, snapshot),
        repo=repo,
        calibration=calibration,
        logit_lens=LogitLens(ctx),
    )
    set_state(state)

    app = FastAPI(title=f"Direction Explorer — {settings.model_name}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for r in all_routers():
        app.include_router(r)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return render_index(ctx, DEFAULT_HARMFUL, DEFAULT_HARMLESS)

    return app
