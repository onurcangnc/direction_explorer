"""FastAPI dependency providers. Wires concrete singletons created by
`create_app()` into route handlers via `Depends()`."""

from __future__ import annotations

from dataclasses import dataclass

from direction_explorer.ablation.service import AblationService
from direction_explorer.config import Settings
from direction_explorer.core.logit_lens import LogitLens
from direction_explorer.core.model_context import ModelContext
from direction_explorer.extractors.registry import ExtractorRegistry
from direction_explorer.persistence.direction_store import (
    CalibrationSet,
    DirectionStore,
)
from direction_explorer.persistence.disk_repository import DiskRepository


@dataclass
class AppState:
    settings: Settings
    ctx: ModelContext
    store: DirectionStore
    registry: ExtractorRegistry
    ablation: AblationService
    repo: DiskRepository
    calibration: CalibrationSet
    logit_lens: LogitLens


# Populated by `create_app()`. Routes call the getters below to read it.
_state: AppState | None = None


def set_state(state: AppState) -> None:
    global _state
    _state = state


def get_state() -> AppState:
    if _state is None:
        raise RuntimeError("AppState not initialized — call create_app() first.")
    return _state


def get_ctx() -> ModelContext:
    return get_state().ctx


def get_store() -> DirectionStore:
    return get_state().store


def get_registry() -> ExtractorRegistry:
    return get_state().registry


def get_ablation() -> AblationService:
    return get_state().ablation


def get_repo() -> DiskRepository:
    return get_state().repo


def get_calibration() -> CalibrationSet:
    return get_state().calibration


def get_logit_lens() -> LogitLens:
    return get_state().logit_lens


def get_settings_dep() -> Settings:
    return get_state().settings
