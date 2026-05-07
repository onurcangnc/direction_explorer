"""direction_explorer — modular refusal-direction explorer for HF causal LMs.

Reference:
    Arditi, A., et al. (2024). "Refusal in Language Models Is Mediated by a
    Single Direction." arXiv:2406.11717
    Piras, F. et al. (2026). SOM-based multi-direction refusal extraction.

Public entry points:
    direction_explorer.api.app.create_app() -> FastAPI
    python -m direction_explorer
"""

__all__ = ["create_app"]

from direction_explorer.api.app import create_app
