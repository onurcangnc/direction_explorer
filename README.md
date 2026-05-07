# direction_explorer

Mechanistic calibration for open-weight causal LMs — reproduces Arditi
et al. (2024) single-direction refusal extraction and Piras et al. (2026)
SOM-based multi-direction extraction, with a three-tab Bootstrap UI for
interactive exploration.

## Run

```bash
python -m direction_explorer
# or with overrides
MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct PORT=8002 python -m direction_explorer
```

Then open http://localhost:8002/.

## Configuration (env vars)

| name                     | default                              |
| ------------------------ | ------------------------------------ |
| `MODEL_NAME`             | `meta-llama/Llama-3.2-3B-Instruct`   |
| `PORT`                   | `8002`                               |
| `HF_TOKEN`               | unset                                |
| `LOAD_IN_8BIT`           | `0`                                  |
| `DEFAULT_MAX_NEW_TOKENS` | `128`                                |
| `MAX_NEW_TOKENS_CAP`     | `512`                                |

Device is auto-detected (CUDA → MPS → CPU).

## Layout

See `CHANGELOG_REFACTOR.md` for the full module map. Top-level shape:

```
direction_explorer/
├── config.py               settings
├── prompts/                default calibration prompts
├── core/                   model context, activations, lens, generation
├── extractors/             mean_diff + SOM (strategy pattern)
├── ablation/               off / partial / full (strategy pattern)
├── persistence/            in-memory store + disk repository
├── api/                    FastAPI routes + create_app factory
└── ui/                     index.html template + renderer
```

## Smoke test

```bash
python3 tests/smoke.py capture --base http://localhost:8002 --out tests/fixtures.json
python3 tests/smoke.py verify  --base http://localhost:8002 --in  tests/fixtures.json
```
