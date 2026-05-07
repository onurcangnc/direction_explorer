"""Project a direction through W_U (lm_head) to find top/bottom vocab tokens.

The score is cosine similarity between the unit direction and each row of
W_U, in [-1, 1]."""

from __future__ import annotations

import torch

from direction_explorer.core.model_context import ModelContext


class LogitLens:
    def __init__(self, ctx: ModelContext):
        self.ctx = ctx

    def project(
        self, direction: torch.Tensor, k: int = 12,
    ) -> tuple[list[dict], list[dict]]:
        """Returns (top, bottom) lists of {token, score} dicts."""
        d = direction.to(torch.float32)
        d = d / (d.norm() + 1e-9)  # d is now unit-norm
        W_U = self.ctx.lm_head.weight.to(torch.float32).to(d.device)
        logits = W_U @ d  # [vocab]
        W_U_norms = W_U.norm(dim=1) + 1e-9
        # d is already unit-normalized, so the denominator is just ‖W_U row‖.
        cosine = logits / W_U_norms

        top_vals, top_idx = cosine.topk(k)
        bot_vals, bot_idx = cosine.topk(k, largest=False)

        tokenizer = self.ctx.tokenizer

        def decode(i):
            try:
                return tokenizer.decode([int(i.item())])
            except Exception:
                return f"<id={int(i.item())}>"

        top = [{"token": decode(i), "score": float(v.item())}
               for i, v in zip(top_idx, top_vals)]
        bot = [{"token": decode(i), "score": float(v.item())}
               for i, v in zip(bot_idx, bot_vals)]
        return top, bot
