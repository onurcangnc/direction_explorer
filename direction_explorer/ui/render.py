"""Render the index template by substituting the four placeholders.

The template lives at `templates/index.html` as a static asset (treated as
opaque — the JS inside is not refactored)."""

from __future__ import annotations

import json
from pathlib import Path

from direction_explorer.core.model_context import ModelContext


_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"


def _load_template() -> str:
    with open(_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def render_index(
    ctx: ModelContext,
    harmful: list[str],
    harmless: list[str],
) -> str:
    default_layer = ctx.n_layers // 2
    max_layer = ctx.n_layers - 1
    return (
        _load_template()
        .replace("__HARMFUL_JSON__", json.dumps(list(harmful)))
        .replace("__HARMLESS_JSON__", json.dumps(list(harmless)))
        .replace("__DEFAULT_LAYER__", str(default_layer))
        .replace("__MAX_LAYER__", str(max_layer))
        .replace("__MODEL_NAME__", ctx.model_name)
    )
