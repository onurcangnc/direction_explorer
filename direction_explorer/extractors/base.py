"""Strategy-pattern base class for direction-extraction methods.

A single `compute()` call returns an `ExtractionResult` containing one or
more `ExtractedDirection`s, each with its own layer-key, direction tensor,
and method-specific metadata. Mean-diff yields 1; SOM yields N (one per
neuron); future SVD will yield K (one per singular vector).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from direction_explorer.core.model_context import ModelContext


@dataclass
class ExtractedDirection:
    layer_key: int | str            # int for canonical, str for "{L}_som_n{i}" etc.
    direction: torch.Tensor         # unit-normalized fp32 CPU tensor [d_model]
    raw_norm: float
    normalized_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    method: str                          # "mean_diff" | "som_md" | …
    layer: int                           # base decoder-layer index
    directions: list[ExtractedDirection]
    summary: dict[str, Any] = field(default_factory=dict)


class ExtractorBase(ABC):
    """Subclasses produce one or more directions for a given layer."""

    method_name: str = "<abstract>"

    def __init__(self, ctx: ModelContext):
        self.ctx = ctx

    @abstractmethod
    def compute(
        self,
        harmful_prompts: list[str],
        harmless_prompts: list[str],
        layer: int,
        **kwargs,
    ) -> ExtractionResult:
        ...
