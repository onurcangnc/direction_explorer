"""Pydantic request schemas. Field names + types are part of the public
API contract — do not rename without bumping the wire version."""

from __future__ import annotations

from pydantic import BaseModel

from direction_explorer.config import get_settings


_DEFAULT_MAX_NEW_TOKENS = get_settings().default_max_new_tokens


class CalibrationRequest(BaseModel):
    harmful_prompts: list[str]
    harmless_prompts: list[str]
    layer: int
    extraction_method: str = "mean_diff"   # "mean_diff" | "som_md"
    n_directions: int = 1                  # reserved for whitened_svd; ignored otherwise
    # SOM-specific (used only when extraction_method == "som_md").
    som_grid_rows: int = 4
    som_grid_cols: int = 4
    som_iterations: int = 10000
    som_learning_rate: float = 0.01
    som_sigma: float = 0.3
    som_seed: int = 42
    replace_canonical: bool = False        # if True, top SOM neuron also writes int-keyed canonical


class AblationRequest(BaseModel):
    prompt: str
    direction_layer: int | str
    extra_direction_layers: list[str] = []  # accepts ints or "{layer}_som_n{i}" / "{layer}_svd{i}"
    mode: str  # "off" | "partial" | "full"
    strength: float = 0.5
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.7


class ComparisonRequest(BaseModel):
    layers: list[str]   # accepts ints or "{layer}_som_n{i}" / "{layer}_svd{i}"
