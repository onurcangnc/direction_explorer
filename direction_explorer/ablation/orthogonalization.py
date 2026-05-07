"""Modified Gram-Schmidt over a list of direction tensors."""

from __future__ import annotations

import torch


def orthonormalize_directions(directions) -> list[torch.Tensor]:
    """Returns a list of mutually orthogonal unit vectors spanning the same
    subspace as `directions`. Order is preserved; nearly-collinear inputs
    are dropped (norm < 1e-6 after subtraction). fp32 on CPU for stability."""
    out: list[torch.Tensor] = []
    for raw in directions:
        v = raw.detach().to("cpu", dtype=torch.float32).clone()
        for u in out:
            v = v - (v @ u) * u
        n = float(v.norm().item())
        if n > 1e-6:
            out.append(v / n)
    return out
