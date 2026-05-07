"""Pure-function projection helpers used by Arditi-style weight
orthogonalization."""

from __future__ import annotations

import torch


def project_out_columns(W: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """For W: [d_out, d_in] whose rows live in d_model space (d_out == d_model),
    make output ⊥ d:
        d · (W @ x) = (d · W) · x = 0  for all x   ⇒   set (d · W) = 0
        W ← W − d ⊗ ((d · W) / ‖d‖²)
    """
    d_norm_sq = (d @ d).clamp_min(1e-12)
    coef = (d @ W) / d_norm_sq        # [d_in]
    return W - d.unsqueeze(-1) * coef.unsqueeze(0)


def project_out_rows(W: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """For W: [N, d_model] where each row is a d_model-dim vector (e.g.
    embed_tokens). Subtract the d-component from every row."""
    d_norm_sq = (d @ d).clamp_min(1e-12)
    coef = (W @ d) / d_norm_sq        # [N]
    return W - coef.unsqueeze(-1) * d.unsqueeze(0)
