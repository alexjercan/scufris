"""Serving the built dashboard bundle.

Mounted at `/` and therefore LAST, after every router: the mount matches
everything, so anything included after it would be unreachable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


class NoCacheStaticFiles(StaticFiles):
    """Serve the SPA bundle with `Cache-Control: no-cache`.

    The bundle filenames are not content-hashed (`agent.js`, `index.html`), so
    without this a browser applies heuristic freshness and can keep running a
    stale bundle for hours without revalidating. `no-cache` forces revalidation
    on every load - the ETag still yields a fast 304 when unchanged, but a
    rebuilt bundle is picked up immediately.
    """

    async def get_response(self, path: str, scope: object) -> Response:
        response = await super().get_response(path, scope)  # type: ignore[arg-type]
        response.headers["Cache-Control"] = "no-cache"
        return response


def mount_web_dist(app: FastAPI, web_dist: Path) -> None:
    """Mount the built bundle at `/`, or warn and serve the API alone.

    A missing bundle is not an error: the API still runs standalone, which is
    what a checkout that has never run `npm run build` gets.
    """
    if not web_dist.is_dir():
        logger.warning(
            "web dist %s not found; serving API only. Build the frontend "
            "(cd web && npm install && npm run build) to serve the dashboard.",
            web_dist,
        )
        return
    app.mount("/", NoCacheStaticFiles(directory=web_dist, html=True), name="web")
