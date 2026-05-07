"""Run with: python -m direction_explorer"""

import uvicorn

from direction_explorer.api.app import create_app
from direction_explorer.config import get_settings


def main() -> None:
    settings = get_settings()
    app = create_app(settings)
    print(f"[RUN] open http://localhost:{settings.port}/")
    uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
