"""Orchestrates the baseline + ablated generation pair.

Capture timing is mode-specific (mirrors the pre-refactor monolith):

  off     — no strategy hooks; capture with ablation_layer_hook=None.
  partial — capture must run AFTER strategy exits, otherwise the hook is
            registered twice (once by the strategy, once by capture) and
            α gets applied twice per layer per forward pass. The strategy
            preserves `capture_hook()` after `_exit()` by design.
  full    — capture must run INSIDE the with-block (weights are mutated
            on enter, restored on exit). hook=None since the weight
            orthogonalization replaces the activation-hook role entirely.
"""

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

        # Ablated run — capture timing differs per mode (see module docstring).
        t1 = time.time()
        try:
            abl_text, abl_ids, abl_prompt_len, abl_tokens, abl_projs = (
                self._run_ablated(
                    mode, formatted, max_new_tokens, temperature,
                    all_directions, strength,
                    primary_layer_int, primary_direction,
                )
            )
        except Exception:
            # Defensive: any escaped strategy state should already be unwound
            # by the with-block, but make doubly sure weights are restored.
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

    def _run_ablated(
        self,
        mode: str,
        formatted: str,
        max_new_tokens: int,
        temperature: float,
        all_directions: list[torch.Tensor],
        strength: float,
        primary_layer_int: int,
        primary_direction: torch.Tensor,
    ):
        ctx = self.ctx
        strategy_cls = AblationStrategyFactory.get(mode)

        if mode == "off":
            # No strategy needed; baseline-style generation + capture w/o hook.
            with strategy_cls(ctx, all_directions, self.snapshot, strength):
                abl_text, abl_ids, abl_prompt_len = hf_generate(
                    ctx, formatted, max_new_tokens, temperature,
                )
            abl_tokens, abl_projs = capture_token_projections(
                ctx, abl_ids, abl_prompt_len, primary_layer_int,
                primary_direction, ablation_layer_hook=None,
            )
            return abl_text, abl_ids, abl_prompt_len, abl_tokens, abl_projs

        if mode == "partial":
            # Hook is registered on every layer for generation, then removed
            # on exit. Capture runs AFTER exit and re-registers the same
            # function once — avoiding the double-fire bug. The strategy
            # keeps `capture_hook()` valid after exit by design.
            with strategy_cls(ctx, all_directions, self.snapshot, strength) as strat:
                abl_text, abl_ids, abl_prompt_len = hf_generate(
                    ctx, formatted, max_new_tokens, temperature,
                )
                saved_hook = strat.capture_hook()
            abl_tokens, abl_projs = capture_token_projections(
                ctx, abl_ids, abl_prompt_len, primary_layer_int,
                primary_direction, ablation_layer_hook=saved_hook,
            )
            return abl_text, abl_ids, abl_prompt_len, abl_tokens, abl_projs

        if mode == "full":
            # Weights are mutated on enter and restored on exit; capture must
            # run INSIDE the with-block to see the orthogonalized weights.
            # No activation hook (the weight projection replaces it).
            with strategy_cls(ctx, all_directions, self.snapshot, strength):
                abl_text, abl_ids, abl_prompt_len = hf_generate(
                    ctx, formatted, max_new_tokens, temperature,
                )
                abl_tokens, abl_projs = capture_token_projections(
                    ctx, abl_ids, abl_prompt_len, primary_layer_int,
                    primary_direction, ablation_layer_hook=None,
                )
            return abl_text, abl_ids, abl_prompt_len, abl_tokens, abl_projs

        raise KeyError(f"unknown mode {mode}")
