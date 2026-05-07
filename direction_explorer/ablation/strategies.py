"""Strategy pattern for ablation modes (off / partial / full).

Each strategy is a context manager so that any mutating side-effect
(forward hooks, weight rewrites) is undone on `__exit__` even on error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type

import torch

from direction_explorer.ablation.orthogonalization import orthonormalize_directions
from direction_explorer.ablation.projection import (
    project_out_columns,
    project_out_rows,
)

if TYPE_CHECKING:
    from direction_explorer.core.model_context import ModelContext
    from direction_explorer.ablation.weight_snapshot import WeightSnapshot


class AblationStrategy(ABC):
    """Context-manager base. `directions` is a list of direction tensors
    already moved to the model device. `enter` arms the strategy; `exit`
    restores any mutated state."""

    name: str = "<abstract>"

    def __init__(
        self,
        ctx: "ModelContext",
        directions: list[torch.Tensor],
        snapshot: "WeightSnapshot",
        strength: float = 1.0,
    ):
        self.ctx = ctx
        self.directions = directions
        self.snapshot = snapshot
        self.strength = float(strength)
        self._capture_hook = None

    def __enter__(self):
        self._enter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._exit()
        return False

    @abstractmethod
    def _enter(self) -> None:
        ...

    @abstractmethod
    def _exit(self) -> None:
        ...

    def capture_hook(self):
        """Return the forward hook function to install on every layer when
        `capture_token_projections` re-runs the model. None if the strategy
        does not require it (off / full mutate weights, not activations)."""
        return self._capture_hook


class OffStrategy(AblationStrategy):
    name = "off"

    def _enter(self) -> None:
        return None

    def _exit(self) -> None:
        return None


class PartialStrategy(AblationStrategy):
    """Subtract α·sum(projection onto each orthonormalized direction) from
    every decoder layer's output residual stream during generation."""

    name = "partial"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._handles: list = []
        self._hook_fn = None

    def _make_hook(self):
        ortho = orthonormalize_directions(self.directions)
        alpha = max(0.0, min(1.0, self.strength))

        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            if not ortho:
                return output
            h_new = h
            for d in ortho:
                d_t = d.to(h_new.dtype).to(h_new.device)
                proj = (h_new @ d_t).unsqueeze(-1) * d_t
                h_new = h_new - alpha * proj
            if is_tuple:
                return (h_new,) + output[1:]
            return h_new

        return hook

    def _enter(self) -> None:
        self._hook_fn = self._make_hook()
        self._capture_hook = self._hook_fn
        self._handles = [
            layer.register_forward_hook(self._hook_fn)
            for layer in self.ctx.layers
        ]

    def _exit(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


class FullStrategy(AblationStrategy):
    """Arditi-style weight orthogonalization: project every direction in
    the orthonormalized basis out of `o_proj`, `down_proj`, and
    `embed_tokens`. Restores from snapshot on exit."""

    name = "full"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weights_orthogonalized = False

    def _apply(self) -> None:
        ortho = orthonormalize_directions(self.directions)
        if not ortho:
            return
        ctx = self.ctx
        with torch.no_grad():
            E = ctx.embed_tokens.weight
            for d in ortho:
                d_t = d.to(E.dtype).to(E.device)
                E.copy_(project_out_rows(E, d_t))
            for layer in ctx.layers:
                o = layer.self_attn.o_proj.weight
                dp = layer.mlp.down_proj.weight
                for d in ortho:
                    d_o = d.to(o.dtype).to(o.device)
                    o.copy_(project_out_columns(o, d_o))
                    d_dp = d.to(dp.dtype).to(dp.device)
                    dp.copy_(project_out_columns(dp, d_dp))
        self._weights_orthogonalized = True

    def _enter(self) -> None:
        self._apply()
        if self.ctx.device == "cuda":
            torch.cuda.empty_cache()

    def _exit(self) -> None:
        if self._weights_orthogonalized:
            self.snapshot.restore()
            self._weights_orthogonalized = False
            if self.ctx.device == "cuda":
                torch.cuda.empty_cache()


class AblationStrategyFactory:
    _strategies: dict[str, Type[AblationStrategy]] = {
        OffStrategy.name: OffStrategy,
        PartialStrategy.name: PartialStrategy,
        FullStrategy.name: FullStrategy,
    }

    @classmethod
    def get(cls, mode: str) -> Type[AblationStrategy]:
        if mode not in cls._strategies:
            raise KeyError(f"unknown mode {mode}")
        return cls._strategies[mode]

    @classmethod
    def modes(cls) -> list[str]:
        return list(cls._strategies)
