"""Mixed int / "{layer}_som_n{i}" / "{layer}_svd{i}" layer-key helpers.

Wire format: layer references arrive on the API as either ints or strings.
In-memory canonical form: int for plain mean_diff, string for SOM/SVD.
"""

from __future__ import annotations

from typing import Any


def parse_layer_key(s) -> int | str:
    """Convert a wire-format layer reference to its in-memory key.

    Canonical (mean_diff) → int. SOM neuron "{layer}_som_n{i}" stays as
    string. Numeric strings ("17") become int. Forward-compatible with
    "{layer}_svd{i}" and other future suffixes."""
    if isinstance(s, int):
        return s
    s = str(s)
    if "_som_n" in s or "_svd" in s:
        return s
    try:
        return int(s)
    except ValueError:
        return s


def computed_layer_sort_key(k) -> tuple:
    """Sort key:  L17 (canonical, kind=0) <  L17_svd* (kind=1) <  L17_som_n* (kind=2)
    Preserves natural ordering by (layer_int, kind, sub_index)."""
    if isinstance(k, int):
        return (k, 0, 0)
    s = str(k)
    if "_som_n" in s:
        layer_part, _, idx_part = s.partition("_som_n")
        try:
            return (int(layer_part), 2, int(idx_part) if idx_part else 0)
        except Exception:
            return (10**9, 2, s)
    if "_svd" in s:
        layer_part, _, idx_part = s.partition("_svd")
        try:
            return (int(layer_part), 1, int(idx_part) if idx_part else 0)
        except Exception:
            return (10**9, 1, s)
    try:
        return (int(s), 0, 0)
    except Exception:
        return (10**9, 9, s)


def layer_label(k, info: dict[str, Any] | None = None) -> str:
    """Display label, e.g. 'L17', 'L17 (SOM n[0,1])', 'L17 (SVD #1)'.

    `info` is the per-direction dict from the store; cached `display_label`
    wins, otherwise we synthesize one from the key shape."""
    info = info or {}
    cached = info.get("display_label")
    if cached:
        return cached
    if isinstance(k, int):
        return f"L{k}"
    s = str(k)
    if "_som_n" in s:
        layer_part, _, idx_part = s.partition("_som_n")
        try:
            i = int(idx_part)
            cols = info.get("som_grid_cols")
            if isinstance(cols, int) and cols > 0:
                return f"L{layer_part} (SOM n[{i // cols},{i % cols}])"
            return f"L{layer_part} (SOM n{idx_part})"
        except Exception:
            return f"L{layer_part} (SOM n{idx_part})"
    if "_svd" in s:
        layer_part, _, idx_part = s.partition("_svd")
        return f"L{layer_part} (SVD #{idx_part})"
    return s


def direction_kind(k, info: dict[str, Any] | None = None) -> str:
    """Returns 'mean_diff' | 'whitened_svd' | 'som_md' based on the key shape."""
    info = info or {}
    if isinstance(k, int):
        return info.get("extraction_method", "mean_diff")
    s = str(k)
    if "_som_n" in s:
        return "som_md"
    if "_svd" in s:
        return "whitened_svd"
    return "mean_diff"


def base_layer_int(k) -> int:
    """Return the underlying decoder-layer index for any key shape."""
    if isinstance(k, int):
        return k
    s = str(k)
    base = s.split("_som_n")[0].split("_svd")[0]
    return int(base)
