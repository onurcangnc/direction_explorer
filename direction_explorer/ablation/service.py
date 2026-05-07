"""Orchestrates the baseline + ablated generation pair."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from direction_explorer.ablation.strategies import AblationStrategyFactory
from direction_explorer.ablation.weight_snapshot import WeightSnapshot
from direction_explorer.core.generation import (
    capture_token_projections,
    hf_generate,
)
from direction_explorer.core.model_context import ModelContext
from direction_explorer.core.prompt_formatting import (
    extract_response_text,
    format_chat_text,
)


@dataclass
class AblationOutput:
    baseline_response: str
    ablated_response: str
    baseline_tokens: list[str]
    ablated_tokens: list[str]
    baseline_token_projections: list[float]
    ablated_token_projections: list[float]
    elapsed_baseline_s: float
    elapsed_ablated_s: float


class AblationService:
    """Owns the WeightSnapshot. Routes call `run()` once per request."""

    def __init__(self, ctx: ModelContext, snapshot: WeightSnapshot):
        self.ctx = ctx
        self.snapshot = snapshot

    def run(
        self,
        prompt: str,
        primary_direction: torch.Tensor,
        all_directions: list[torch.Tensor],
        primary_layer_int: int,
        mode: str,
        strength: float,
        max_new_tokens: int,
        temperature: float,
    ) -> AblationOutput:
        ctx = self.ctx
        formatted = format_chat_text(ctx.tokenizer, prompt)

        if ctx.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Defensive: always start from clean weights.
        self.snapshot.restore()
        if ctx.device == "cuda":
            torch.cuda.empty_cache()

        # Baseline run (no ablation).
        t0 = time.time()
        base_text, base_ids, prompt_len = hf_generate(
            ctx, formatted, max_new_tokens, temperature,
        )
        base_response = extract_response_text(
            ctx.tokenizer.decode(base_ids, skip_special_tokens=False), formatted,
        ) or base_text
        base_tokens, base_projs = capture_token_projections(
            ctx, base_ids, prompt_len, primary_layer_int, primary_direction,
            ablation_layer_hook=None,
        )
        elapsed_baseline = round(time.time() - t0, 2)
        if ctx.device == "cuda":
            torch.cuda.empty_cache()

        # Ablated run.
        t1 = time.time()
        strategy_cls = AblationStrategyFactory.get(mode)
        try:
            with strategy_cls(ctx, all_directions, self.snapshot, strength) as strat:
                abl_text, abl_ids, abl_prompt_len = hf_generate(
                    ctx, formatted, max_new_tokens, temperature,
                )
                abl_tokens, abl_projs = capture_token_projections(
                    ctx, abl_ids, abl_prompt_len, primary_layer_int,
                    primary_direction,
                    ablation_layer_hook=strat.capture_hook(),
                )
        except Exception:
            # Strategy.__exit__ should have already restored, but be defensive.
            self.snapshot.restore()
            if ctx.device == "cuda":
                torch.cuda.empty_cache()
            raise

        abl_response = extract_response_text(
            ctx.tokenizer.decode(abl_ids, skip_special_tokens=False), formatted,
        ) or abl_text
        elapsed_ablated = round(time.time() - t1, 2)

        return AblationOutput(
            baseline_response=base_response,
            ablated_response=abl_response,
            baseline_tokens=base_tokens,
            ablated_tokens=abl_tokens,
            baseline_token_projections=base_projs,
            ablated_token_projections=abl_projs,
            elapsed_baseline_s=elapsed_baseline,
            elapsed_ablated_s=elapsed_ablated,
        )
