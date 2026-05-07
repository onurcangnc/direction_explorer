"""Registry/factory for extraction strategies. Adding a new extractor =
register a new class here, no edits to existing ones (OCP)."""

from __future__ import annotations

from typing import Type

from direction_explorer.core.model_context import ModelContext
from direction_explorer.extractors.base import ExtractorBase
from direction_explorer.extractors.mean_diff import MeanDiffExtractor
from direction_explorer.extractors.som import SOMExtractor


class ExtractorRegistry:
    def __init__(self):
        self._registered: dict[str, Type[ExtractorBase]] = {}

    def register(self, extractor_cls: Type[ExtractorBase]) -> None:
        name = extractor_cls.method_name
        if not name or name == "<abstract>":
            raise ValueError("extractor_cls.method_name must be set")
        self._registered[name] = extractor_cls

    def get(self, method: str, ctx: ModelContext) -> ExtractorBase:
        if method not in self._registered:
            raise KeyError(
                f"unknown extraction_method '{method}' "
                f"(allowed: {sorted(self._registered)})"
            )
        return self._registered[method](ctx)

    def methods(self) -> list[str]:
        return sorted(self._registered)


def default_registry() -> ExtractorRegistry:
    r = ExtractorRegistry()
    r.register(MeanDiffExtractor)
    r.register(SOMExtractor)
    return r
