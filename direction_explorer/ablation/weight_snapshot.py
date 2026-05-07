"""Snapshot of the writable weights touched by full-ablation, kept on CPU.

The full-ablation strategy mutates `self_attn.o_proj`, `mlp.down_proj`, and
`embed_tokens` in-place. WeightSnapshot lets us restore them after a run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from direction_explorer.core.model_context import ModelContext


class WeightSnapshot:
    def __init__(self, ctx: "ModelContext"):
        self.ctx = ctx
        print("[SNAPSHOT] Cloning attention o_proj, mlp down_proj, embedding weights (to CPU)...")
        self._o_proj = [
            layer.self_attn.o_proj.weight.detach().to("cpu", copy=True)
            for layer in ctx.layers
        ]
        self._down_proj = [
            layer.mlp.down_proj.weight.detach().to("cpu", copy=True)
            for layer in ctx.layers
        ]
        self._embed_tokens = ctx.embed_tokens.weight.detach().to("cpu", copy=True)
        if ctx.device == "cuda":
            print(
                f"[SNAPSHOT] VRAM allocated after snapshot: "
                f"{torch.cuda.memory_allocated()/1e9:.2f}GB"
            )
        print("[SNAPSHOT] Done.")

    def restore(self) -> None:
        ctx = self.ctx
        with torch.no_grad():
            for i, layer in enumerate(ctx.layers):
                layer.self_attn.o_proj.weight.copy_(self._o_proj[i])
                layer.mlp.down_proj.weight.copy_(self._down_proj[i])
            ctx.embed_tokens.weight.copy_(self._embed_tokens)
