# Refactor — modular architecture

The single-file `direction_explorer.py` (~2,276 lines) has been split into
the `direction_explorer/` package. Behavior is preserved verbatim:
HTTP contracts, on-disk filenames, JSON schema, tensor `.pt` format, env
vars, and UI all unchanged.

## Run

Old:    `python3 direction_explorer.py`
New:    `python -m direction_explorer`

Same env vars (`MODEL_NAME`, `PORT`, `HF_TOKEN`, `LOAD_IN_8BIT`,
`DEFAULT_MAX_NEW_TOKENS`, `MAX_NEW_TOKENS_CAP`).

## Module map (one-line responsibility per file)

### Top-level
- `direction_explorer/__init__.py` — re-exports `create_app`.
- `direction_explorer/__main__.py` — `python -m direction_explorer` entry; loads settings, creates app, runs uvicorn.
- `direction_explorer/config.py` — frozen `Settings` dataclass; reads env vars; replaces all scattered `os.environ` lookups.

### Defaults
- `direction_explorer/prompts/defaults.py` — `DEFAULT_HARMFUL` / `DEFAULT_HARMLESS` prompt lists (verbatim).

### Core (model + activations + lens + generation)
- `direction_explorer/core/model_context.py` — frozen `ModelContext` bundling model+tokenizer+device+dims; replaces module-level globals.
- `direction_explorer/core/model_loader.py` — `load_model_context(settings) -> ModelContext` (preserves CUDA/8-bit/MPS/CPU paths).
- `direction_explorer/core/prompt_formatting.py` — chat-template formatting + response-text extraction.
- `direction_explorer/core/activations.py` — single-layer mean residual + per-prompt activation collection (forward-hook based).
- `direction_explorer/core/logit_lens.py` — `LogitLens` service projecting a unit direction through `lm_head`. Drops the redundant `(d.norm() + 1e-9)` factor since `d` is already unit-normalized.
- `direction_explorer/core/generation.py` — `hf_generate` and `capture_token_projections`.

### Extractors (Strategy + Registry)
- `direction_explorer/extractors/base.py` — `ExtractorBase` ABC + `ExtractedDirection` / `ExtractionResult` dataclasses (1-or-N directions per call).
- `direction_explorer/extractors/mean_diff.py` — `MeanDiffExtractor` (Arditi 2024).
- `direction_explorer/extractors/som.py` — `SOMExtractor` + `SimpleSOM` helper + `compute_som_directions` (Piras 2026); SOM math is byte-for-byte preserved.
- `direction_explorer/extractors/registry.py` — `ExtractorRegistry` factory; `default_registry()` registers `mean_diff` and `som_md`.

### Ablation (Strategy + Service)
- `direction_explorer/ablation/projection.py` — pure-function `project_out_columns` / `project_out_rows`.
- `direction_explorer/ablation/orthogonalization.py` — `orthonormalize_directions` (modified Gram-Schmidt).
- `direction_explorer/ablation/weight_snapshot.py` — `WeightSnapshot` clones `o_proj`, `down_proj`, `embed_tokens` to CPU and restores on demand.
- `direction_explorer/ablation/strategies.py` — `OffStrategy` / `PartialStrategy` / `FullStrategy` as context managers; `AblationStrategyFactory.get(mode)`.
- `direction_explorer/ablation/service.py` — `AblationService` orchestrates the baseline + ablated runs.

### Persistence (Repository)
- `direction_explorer/persistence/layer_keys.py` — `parse_layer_key`, `computed_layer_sort_key`, `layer_label`, `direction_kind`, `base_layer_int`.
- `direction_explorer/persistence/direction_store.py` — abstract `DirectionStore` + `InMemoryDirectionStore`; `CalibrationSet` for harmful/harmless/id triple.
- `direction_explorer/persistence/disk_repository.py` — `DiskRepository.persist()` and `.load_into(store, calibration)`; same filename patterns and JSON schema as before.

### API
- `direction_explorer/api/schemas.py` — `CalibrationRequest`, `AblationRequest`, `ComparisonRequest` (pydantic) — field names + types match the pre-refactor contract.
- `direction_explorer/api/deps.py` — `AppState` + FastAPI dependency providers (`get_ctx`, `get_store`, `get_registry`, `get_ablation`, …).
- `direction_explorer/api/routes/state.py` — `GET /state`.
- `direction_explorer/api/routes/calibration.py` — `POST /calibration/compute` (mean_diff + som_md responders).
- `direction_explorer/api/routes/ablation.py` — `POST /ablation/generate`.
- `direction_explorer/api/routes/comparison.py` — `POST /comparison/analyze`.
- `direction_explorer/api/app.py` — `create_app(settings)` factory wires everything; the only place that constructs singletons.

### UI
- `direction_explorer/ui/templates/index.html` — the giant Bootstrap+JS page, lifted out as a static asset (no edits to the JS).
- `direction_explorer/ui/render.py` — `render_index(ctx, harmful, harmless)` substitutes the four placeholders.

## Bugs rolled in during the refactor

1. **`int(req.direction_layer)` crash on Piras keys** — already fixed in the
   uncommitted modifications to `direction_explorer.py` and preserved here.
   `parse_layer_key` + `base_layer_int` are now used everywhere.
2. **`_resolve_ablation_layers` docstring** — old wording claimed primary
   was always int. New version (in `routes/ablation.py`) accurately describes
   that primary may be int (mean_diff) or str (SOM/SVD).
3. **`logit_lens` redundant denominator factor** — `cosine = logits / W_U_norms`
   instead of `logits / (W_U_norms * (d.norm() + 1e-9))`; `d` is unit-normalized
   one line earlier so the extra factor was always 1.

## SOLID pattern application notes

- **SRP** — `model_loader` knows nothing about extractors; `extractors/som.py`
  knows nothing about FastAPI; `routes/state.py` knows nothing about disk format.
- **OCP** — adding a new extraction method = new file under `extractors/` +
  one line in `default_registry()`. No edits to `mean_diff.py` or `som.py`.
- **LSP** — every `ExtractorBase` subclass returns an `ExtractionResult`;
  the calibration responders branch on `method` only at the boundary.
- **ISP** — `ExtractorBase` does not know about ablation; `AblationStrategy`
  does not know about extraction; routes consume the abstractions they need.
- **DIP** — routes depend on `DirectionStore` (abstract), not on a global dict.

## Verification

`tests/smoke.py` is a fixture-based parity harness:

```bash
# Phase 1: against the pre-refactor monolith
python3 direction_explorer.py
python3 tests/smoke.py capture --base http://localhost:8002 --out tests/fixtures.json

# Phase 2: against the new package
python -m direction_explorer
python3 tests/smoke.py verify --base http://localhost:8002 --in tests/fixtures.json
```

For tensor-level bit equality (mean_diff at fixed seed=42, identical
prompts), compare `results/<slug>_L<NNN>_v<ID>.pt` files directly with
`torch.equal()`.
