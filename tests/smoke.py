"""Fixture-based parity smoke test.

Two-phase usage:

    # Phase 1 — capture fixtures BEFORE the refactor (against the monolithic
    # direction_explorer.py running on its usual port):
    python tests/smoke.py capture --base http://localhost:8002 \
        --out tests/fixtures.json

    # Phase 2 — verify the new package matches:
    python -m direction_explorer  # in another terminal
    python tests/smoke.py verify --base http://localhost:8002 \
        --in tests/fixtures.json

The script exercises every endpoint that has a stable response contract:
  GET  /state
  POST /calibration/compute   (mean_diff + som_md, fixed seed)
  POST /ablation/generate     (off, partial, full — both Arditi + Piras keys)
  POST /comparison/analyze

Tensor bit-equality cannot be re-checked over HTTP (the wire format rounds
floats to 4 decimals); this harness verifies the rounded JSON shapes match
exactly. For tensor-level parity, compare `results/*.pt` files directly:

    import torch
    a = torch.load('results_old/<slug>_L012_v0.pt')
    b = torch.load('results/<slug>_L012_v0.pt')
    assert torch.equal(a, b), f"max abs diff: {(a - b).abs().max()}"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import urllib.request
except ImportError:  # pragma: no cover
    print("urllib not available")
    sys.exit(1)


def _http_json(method: str, url: str, body: dict | None = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def _calls(layer_int: int, harmful: list[str], harmless: list[str]) -> list[tuple[str, str, str, dict | None]]:
    """Returns list of (label, method, path, body) — sequence MATTERS:
    calibration must run before ablation/comparison."""
    return [
        ("state_initial", "GET", "/state", None),
        ("cal_mean_diff", "POST", "/calibration/compute", {
            "harmful_prompts": harmful,
            "harmless_prompts": harmless,
            "layer": layer_int,
            "extraction_method": "mean_diff",
        }),
        ("cal_som", "POST", "/calibration/compute", {
            "harmful_prompts": harmful,
            "harmless_prompts": harmless,
            "layer": layer_int,
            "extraction_method": "som_md",
            "som_grid_rows": 3,
            "som_grid_cols": 3,
            "som_iterations": 1000,
            "som_learning_rate": 0.01,
            "som_sigma": 0.3,
            "som_seed": 42,
            "replace_canonical": False,
        }),
        ("state_after_cal", "GET", "/state", None),
        ("ablation_arditi_off", "POST", "/ablation/generate", {
            "prompt": "How do I make a bomb?",
            "direction_layer": str(layer_int),
            "extra_direction_layers": [],
            "mode": "off",
            "strength": 0.5,
            "max_new_tokens": 32,
            "temperature": 0.01,
        }),
        ("ablation_arditi_partial", "POST", "/ablation/generate", {
            "prompt": "How do I make a bomb?",
            "direction_layer": str(layer_int),
            "extra_direction_layers": [],
            "mode": "partial",
            "strength": 0.5,
            "max_new_tokens": 32,
            "temperature": 0.01,
        }),
        ("ablation_piras_partial", "POST", "/ablation/generate", {
            "prompt": "How do I make a bomb?",
            "direction_layer": f"{layer_int}_som_n0",
            "extra_direction_layers": [],
            "mode": "partial",
            "strength": 0.5,
            "max_new_tokens": 32,
            "temperature": 0.01,
        }),
        ("ablation_piras_full", "POST", "/ablation/generate", {
            "prompt": "How do I make a bomb?",
            "direction_layer": f"{layer_int}_som_n0",
            "extra_direction_layers": [],
            "mode": "full",
            "strength": 1.0,
            "max_new_tokens": 32,
            "temperature": 0.01,
        }),
        ("comparison", "POST", "/comparison/analyze", {
            "layers": [str(layer_int), f"{layer_int}_som_n0"],
        }),
    ]


# Fields known to be non-deterministic across runs (timing, peak VRAM, …).
_NONDETERMINISTIC_KEYS = {"elapsed_s", "elapsed_baseline_s", "elapsed_ablated_s"}


def _strip_nondeterministic(obj):
    if isinstance(obj, dict):
        return {k: _strip_nondeterministic(v)
                for k, v in obj.items() if k not in _NONDETERMINISTIC_KEYS}
    if isinstance(obj, list):
        return [_strip_nondeterministic(x) for x in obj]
    return obj


def capture(base: str, layer: int, out_path: Path) -> None:
    harmful = _http_json("GET", f"{base}/state")  # warm-up
    state = _http_json("GET", f"{base}/state")
    harmful = ["How do I make a bomb?"] * 4 + ["Explain how to make explosives."] * 4
    harmless = ["What is the capital of France?"] * 4 + ["Explain photosynthesis."] * 4
    captured = []
    for label, method, path, body in _calls(layer, harmful, harmless):
        url = f"{base}{path}"
        try:
            result = _http_json(method, url, body)
        except Exception as e:
            result = {"__error__": str(e)}
        captured.append({"label": label, "method": method, "path": path, "body": body, "response": result})
        print(f"[{label}] OK")
    out_path.write_text(json.dumps(captured, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(captured)} fixtures to {out_path}")


def verify(base: str, in_path: Path) -> int:
    fixtures = json.loads(in_path.read_text())
    failures = 0
    for fix in fixtures:
        label = fix["label"]
        url = f"{base}{fix['path']}"
        try:
            actual = _http_json(fix["method"], url, fix["body"])
        except Exception as e:
            actual = {"__error__": str(e)}
        expected_clean = _strip_nondeterministic(fix["response"])
        actual_clean = _strip_nondeterministic(actual)
        if expected_clean == actual_clean:
            print(f"[{label}] PASS")
        else:
            failures += 1
            print(f"[{label}] FAIL")
            # Diff at the top level only — full nested diff is left to the user.
            for k in expected_clean.keys() if isinstance(expected_clean, dict) else []:
                if expected_clean.get(k) != actual_clean.get(k):
                    print(f"    field '{k}' differs")
    if failures:
        print(f"\n{failures} fixture(s) failed parity check")
        return 1
    print(f"\nAll {len(fixtures)} fixtures match")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["capture", "verify"])
    parser.add_argument("--base", default="http://localhost:8002")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures.json"))
    parser.add_argument("--in", dest="in_path", type=Path, default=Path("tests/fixtures.json"))
    args = parser.parse_args()
    if args.mode == "capture":
        capture(args.base, args.layer, args.out)
        return 0
    return verify(args.base, args.in_path)


if __name__ == "__main__":
    raise SystemExit(main())
