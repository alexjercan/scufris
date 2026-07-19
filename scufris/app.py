"""The Scufris FastAPI application.

Serves a read-only JSON stats API and, when built, the static dashboard bundle.
The stats collector is injected so tests can supply a fake; production uses the
psutil-backed collector.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .metrics import Collector, HostStats, PsutilCollector

logger = logging.getLogger(__name__)


def create_app(
    collector: Collector | None = None, settings: Settings | None = None
) -> FastAPI:
    settings = settings or Settings()
    collector = collector or PsutilCollector()

    app = FastAPI(title="Scufris", description="Scuffed Jarvis host dashboard")

    @app.get("/api/stats")
    def get_stats() -> HostStats:
        """Return a fresh read-only snapshot of host metrics."""
        return collector.sample()

    @app.get("/api/config")
    def get_config() -> dict[str, float]:
        """Client-facing knobs (currently just the poll interval)."""
        return {"poll_seconds": settings.poll_seconds}

    # Mount the built dashboard LAST so the /api routes above take precedence;
    # everything else falls through to the static bundle. Skipped (with a hint)
    # until the frontend has been built, so the API still runs standalone.
    if settings.web_dist.is_dir():
        app.mount("/", StaticFiles(directory=settings.web_dist, html=True), name="web")
    else:
        logger.warning(
            "web dist %s not found; serving API only. Build the frontend "
            "(cd web && npm install && npm run build) to serve the dashboard.",
            settings.web_dist,
        )

    return app


def main() -> None:
    """Console entry point: launch the app with uvicorn."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    uvicorn.run(create_app(settings=settings), host=settings.host, port=settings.port)
