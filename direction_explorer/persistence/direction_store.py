"""Repository pattern for cached direction state.

Endpoints depend on the abstract `DirectionStore`, not on a global dict.
Single in-memory implementation backed by a dict; disk durability is
handled separately by `DiskRepository` so the store can stay small + pure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from direction_explorer.persistence.layer_keys import computed_layer_sort_key


class DirectionStore(ABC):
    """Mapping layer-key -> entry dict.

    An entry contains at minimum:
        direction (Tensor), raw_norm (float), normalized_score (float),
        top_tokens (list), bottom_tokens (list), calibration_set_id (int),
        model_name (str), n_layers (int), d_model (int),
        extraction_method (str), display_label (str | None).
    SOM neurons additionally carry: lattice_position, neuron_index,
        cluster_size, cluster_share, cluster_tightness, som_grid_rows,
        som_grid_cols.
    """

    @abstractmethod
    def get(self, key) -> dict | None: ...

    @abstractmethod
    def put(self, key, entry: dict) -> None: ...

    @abstractmethod
    def keys(self) -> list: ...

    @abstractmethod
    def items_sorted(self) -> list[tuple[Any, dict]]: ...

    @abstractmethod
    def exists(self, key) -> bool: ...


class InMemoryDirectionStore(DirectionStore):
    def __init__(self):
        self._data: dict = {}

    def get(self, key) -> dict | None:
        return self._data.get(key)

    def put(self, key, entry: dict) -> None:
        self._data[key] = entry

    def keys(self) -> list:
        return sorted(self._data.keys(), key=computed_layer_sort_key)

    def items_sorted(self) -> list[tuple[Any, dict]]:
        return sorted(self._data.items(), key=lambda kv: computed_layer_sort_key(kv[0]))

    def exists(self, key) -> bool:
        return key in self._data

    # --- convenience for routes that still want raw access ---
    def __contains__(self, key) -> bool:
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]


class CalibrationSet:
    """The current 'calibration set' the user is iterating on. Bumps `id`
    when the prompts change so cached directions can be tagged."""

    def __init__(self, harmful: list[str], harmless: list[str], set_id: int = 0):
        self.harmful = list(harmful)
        self.harmless = list(harmless)
        self.id = int(set_id)

    def replace_if_changed(self, harmful: list[str], harmless: list[str]) -> int:
        if harmful != self.harmful or harmless != self.harmless:
            self.harmful = list(harmful)
            self.harmless = list(harmless)
            self.id += 1
        return self.id
