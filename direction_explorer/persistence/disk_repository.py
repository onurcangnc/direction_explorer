"""Disk-backed direction storage.

Filename patterns (must remain stable so old `results/` files keep loading):
    canonical mean_diff: {slug}_L{NNN}_v{ID}.json|.pt
    SOM neuron #i:       {slug}_L{NNN}_som_n{II}_v{ID}.json|.pt   (II zero-padded to 2)
    whitened SVD #i:     {slug}_L{NNN}_svd{i}_v{ID}.json|.pt      (forward-compat)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from direction_explorer.config import Settings
from direction_explorer.persistence.direction_store import (
    CalibrationSet,
    DirectionStore,
)
from direction_explorer.persistence.layer_keys import (
    computed_layer_sort_key,
    layer_label,
)


_SOM_RE = re.compile(r".*_L(\d{3})_som_n(\d+)_v(\d+)\.json$")
_SVD_RE = re.compile(r".*_L(\d{3})_svd(\d+)_v(\d+)\.json$")


class DiskRepository:
    def __init__(
        self,
        settings: Settings,
        n_layers: int,
        d_model: int,
    ):
        self.settings = settings
        self.results_dir: Path = settings.results_dir
        self.n_layers = n_layers
        self.d_model = d_model
        self.model_name = settings.model_name
        self.slug = settings.safe_model_slug

    # ── stem builder ─────────────────────────────────────────────────────
    def _stem(self, layer_key, set_id: int) -> tuple[str, int]:
        if isinstance(layer_key, int):
            layer_int = int(layer_key)
            return f"{self.slug}_L{layer_int:03d}_v{set_id}", layer_int
        s = str(layer_key)
        if "_som_n" in s:
            lp, _, ip = s.partition("_som_n")
            layer_int = int(lp)
            return f"{self.slug}_L{layer_int:03d}_som_n{int(ip):02d}_v{set_id}", layer_int
        if "_svd" in s:
            lp, _, ip = s.partition("_svd")
            layer_int = int(lp)
            return f"{self.slug}_L{layer_int:03d}_svd{int(ip)}_v{set_id}", layer_int
        layer_int = int(s)
        return f"{self.slug}_L{layer_int:03d}_v{set_id}", layer_int

    def persist(
        self,
        layer_key,
        direction: torch.Tensor,
        raw_norm: float,
        normalized_score: float,
        top_tokens: list,
        bottom_tokens: list,
        set_id: int,
        extraction_method: str = "mean_diff",
        extra_meta: dict | None = None,
    ) -> None:
        try:
            stem, layer_int = self._stem(layer_key, set_id)
            meta_path = self.results_dir / f"{stem}.json"
            tensor_path = meta_path.with_suffix(".pt")
            payload = {
                "model": self.model_name,
                "layer": layer_int,
                "layer_key": str(layer_key),
                "raw_norm": raw_norm,
                "normalized_score": normalized_score,
                "top_tokens": top_tokens,
                "bottom_tokens": bottom_tokens,
                "calibration_set_id": set_id,
                "n_layers": self.n_layers,
                "d_model": self.d_model,
                "extraction_method": extraction_method,
            }
            if extra_meta:
                payload.update(extra_meta)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            torch.save(direction.detach().cpu(), tensor_path)
        except Exception as e:
            print(f"[PERSIST] Warning: could not save {layer_key}: {e}")

    def load_into(
        self,
        store: DirectionStore,
        calibration: CalibrationSet,
    ) -> int:
        """Load all saved directions matching this model. Returns count
        loaded. Mutates `store` and bumps `calibration.id` if a higher
        set_id is observed on disk."""
        candidates = sorted(self.results_dir.glob(f"{self.slug}_L*.json"))
        if not candidates:
            return 0

        best_per_key: dict = {}
        for meta_path in candidates:
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("model") != self.model_name:
                    continue
                if meta.get("n_layers") != self.n_layers or meta.get("d_model") != self.d_model:
                    continue
                layer_int = int(meta["layer"])
                set_id = int(meta.get("calibration_set_id", 0))
                name = meta_path.name
                if _SOM_RE.match(name) is not None:
                    idx = int(_SOM_RE.match(name).group(2))
                    key = f"{layer_int}_som_n{idx}"
                elif _SVD_RE.match(name) is not None:
                    idx = int(_SVD_RE.match(name).group(2))
                    key = f"{layer_int}_svd{idx}"
                else:
                    key = layer_int
                cur = best_per_key.get(key)
                if cur is None or set_id > cur[0]:
                    best_per_key[key] = (set_id, meta_path)
            except Exception as e:
                print(f"[PERSIST] Skipping {meta_path.name}: {e}")
                continue

        loaded = 0
        max_set_id = 0
        for key, (set_id, meta_path) in sorted(
            best_per_key.items(), key=lambda kv: computed_layer_sort_key(kv[0]),
        ):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                tensor_path = meta_path.with_suffix(".pt")
                if not tensor_path.exists():
                    print(f"[PERSIST] Missing tensor for {key}: {tensor_path.name}")
                    continue
                direction = torch.load(tensor_path, map_location="cpu")
                if direction.shape[-1] != self.d_model:
                    print(f"[PERSIST] Shape mismatch {key}: {tuple(direction.shape)}")
                    continue
                method = meta.get("extraction_method", "mean_diff")
                entry = {
                    "direction": direction.detach().to("cpu"),
                    "raw_norm": float(meta["raw_norm"]),
                    "normalized_score": float(meta["normalized_score"]),
                    "top_tokens": meta.get("top_tokens", []),
                    "bottom_tokens": meta.get("bottom_tokens", []),
                    "calibration_set_id": set_id,
                    "model_name": self.model_name,
                    "n_layers": self.n_layers,
                    "d_model": self.d_model,
                    "extraction_method": method,
                }
                for k_meta in ("lattice_position", "neuron_index", "cluster_size",
                               "cluster_share", "cluster_tightness",
                               "som_grid_rows", "som_grid_cols"):
                    if k_meta in meta:
                        entry[k_meta] = meta[k_meta]
                # display_label is computed from key + entry; cache it.
                entry["display_label"] = layer_label(key, entry)
                store.put(key, entry)
                loaded += 1
                max_set_id = max(max_set_id, set_id)
            except Exception as e:
                print(f"[PERSIST] Failed to load {key}: {e}")
                continue

        if max_set_id > calibration.id:
            calibration.id = max_set_id
        return loaded
