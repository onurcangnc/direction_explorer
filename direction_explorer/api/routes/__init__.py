from fastapi import APIRouter

from direction_explorer.api.routes.ablation import router as ablation_router
from direction_explorer.api.routes.calibration import router as calibration_router
from direction_explorer.api.routes.comparison import router as comparison_router
from direction_explorer.api.routes.state import router as state_router


def all_routers() -> list[APIRouter]:
    return [state_router, calibration_router, ablation_router, comparison_router]
