"""Arditi 2024 mean-difference extraction.

direction = unit(μ_harmful − μ_harmless) at the chosen layer.
"""

from __future__ import annotations

import torch

from direction_explorer.core.activations import compute_single_layer_mean
from direction_explorer.extractors.base import (
    ExtractedDirection,
    ExtractionResult,
    ExtractorBase,
)


class MeanDiffExtractor(ExtractorBase):
    method_name = "mean_diff"

    def compute(
        self,
        harmful_prompts: list[str],
        harmless_prompts: list[str],
        layer: int,
        **kwargs,
    ) -> ExtractionResult:
        ctx = self.ctx
        mean_h = compute_single_layer_mean(ctx, harmful_prompts, layer)
        mean_n = compute_single_layer_mean(ctx, harmless_prompts, layer)
        mean_diff = mean_h - mean_n
        raw_norm = float(mean_diff.norm().item())
        direction = mean_diff / (mean_diff.norm() + 1e-9)
        baseline = (mean_h.norm().item() + mean_n.norm().item()) / 2
        normalized_score = raw_norm / (baseline + 1e-9)

        ed = ExtractedDirection(
            layer_key=layer,
            direction=direction.detach().to("cpu"),
            raw_norm=raw_norm,
            normalized_score=normalized_score,
            metadata={"display_label": f"L{layer}"},
        )
        return ExtractionResult(
            method=self.method_name,
            layer=layer,
            directions=[ed],
        )
