"""ModelContext: bundle of model + tokenizer + device + dimensions.

Replaces the module-level globals (`model`, `tokenizer`, `DEVICE`, `D_MODEL`,
`N_LAYERS`). One instance per app, passed explicitly into anything that needs
it (extractors, ablation strategies, generation helpers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class ModelContext:
    model: "PreTrainedModel"
    tokenizer: "PreTrainedTokenizerBase"
    device: str
    dtype: torch.dtype
    d_model: int
    n_layers: int
    model_name: str

    @property
    def layers(self):
        return self.model.model.layers

    @property
    def embed_tokens(self):
        return self.model.model.embed_tokens

    @property
    def lm_head(self):
        return self.model.lm_head
