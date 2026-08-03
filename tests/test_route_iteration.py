"""`iter_routes` fails CLOSED, falsified rather than assumed.

`iter_routes` exists because `include_router` started producing a node the old
`isinstance(route, Route)` idiom skipped silently, so every guard that proves a
property of "every route" quietly covered less while still reporting green. Its
defence against that is to refuse an unrecognized node instead of skipping it.

A guard is not a guard until it has been falsified, and these three cases are
that falsification: each one is red against a tree with its own `raise` removed.
The happy path - that a bare included router IS walked - is covered where it
matters, by `tests/test_route_contract.py` and the auth boundary sweep.
"""

from __future__ import annotations

import pytest
from fastapi import APIRouter, FastAPI
from starlette.routing import BaseRoute, WebSocketRoute

from scufris.api.routes import iter_routes


def test_a_websocket_route_is_refused_rather_than_skipped() -> None:
    """A websocket is refused, so the boundary sweep cannot lose it.

    A skipped websocket is the fail-open hole in miniature: `BaseHTTPMiddleware`
    never sees a websocket scope, so an endpoint dropped from the sweep is one no
    guard covers at all. Refusing means the first websocket added to this app
    lands with the sweep as the thing that stops it.
    """
    app = FastAPI()
    app.routes.append(WebSocketRoute("/ws", endpoint=lambda websocket: None))

    with pytest.raises(TypeError, match="does not recognize"):
        list(iter_routes(app))


def test_an_unrecognized_route_node_is_refused() -> None:
    """Any other unknown node raises rather than shrinking the surface."""

    class Exotic(BaseRoute):
        pass

    app = FastAPI()
    app.routes.append(Exotic())

    with pytest.raises(TypeError, match="does not recognize"):
        list(iter_routes(app))


def test_a_router_included_with_a_prefix_is_refused() -> None:
    """Callers read `route.path`; under an include-time prefix it is a lie."""
    app = FastAPI()
    app.include_router(APIRouter(), prefix="/x")

    with pytest.raises(ValueError, match="prefix or tags"):
        list(iter_routes(app))
