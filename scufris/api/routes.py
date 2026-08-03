"""Walking the app's real route surface, through included routers.

FastAPI 0.139 changed ``include_router``: it no longer copies each route onto
the app, it appends ONE opaque ``_IncludedRouter`` node that resolves the
router's routes lazily at request time. So the moment a route moves onto a
router, ``for route in app.routes: if isinstance(route, Route)`` stops seeing it
- silently.

Anything that derives the app's real surface - the OpenAPI tag assignment, and
every guard that proves a property of "every route" - must walk routers too, or
it quietly covers less while still passing. ``iter_routes`` is the one
replacement for that idiom, so a router added later is covered by existing
rather than by remembering.

It fails CLOSED. A node it does not recognize raises rather than being skipped,
because a silent skip is exactly the failure the function exists to prevent.
"""

from __future__ import annotations

from typing import Any, Iterator

from fastapi.routing import APIRoute
from starlette.routing import Mount, Route


def iter_routes(target: Any) -> Iterator[Route]:
    """Every ``Route`` reachable from an app or router, routers included.

    Yields plain Starlette routes as well as ``APIRoute``s: ``/openapi.json``,
    ``/docs`` and ``/redoc`` are plain routes, and the auth boundary sweep would
    otherwise leave them outside the surface it proves is gated.

    Refuses to descend into a router included with a ``prefix`` or with
    ``tags``: callers read ``route.path`` and ``route.tags`` off what comes back,
    and under an include-time prefix or tag those attributes are no longer what
    the served route answers to. Every router in this app is included bare - the
    paths are absolute on the router itself - so this raises only if that
    changes, which is exactly when the callers need to be revisited.

    Raises on any other node as well. The whole reason this function exists is
    that ``include_router`` started producing a node the old idiom skipped
    silently, so skipping an unrecognized one would rebuild that bug here: the
    guards would go back to covering less while still reporting green.
    """
    for route in getattr(target, "router", target).routes:
        if isinstance(route, Route):
            yield route
            continue
        nested = getattr(route, "original_router", None)
        if nested is not None:
            context = route.include_context
            if context.prefix or context.tags:
                raise ValueError(
                    f"{nested!r} is included with a prefix or tags "
                    f"({context.prefix!r}, {context.tags!r}); the route objects "
                    "it holds no longer carry the path and tags it is served "
                    "under, so every caller of iter_routes would report the "
                    "wrong surface."
                )
            yield from iter_routes(nested)
            continue
        # A mounted sub-application (the static dist) has no HTTP route table to
        # contribute, and every caller asks about HTTP routes. Skipped
        # deliberately, and named, so that the skip is a decision rather than a
        # fallthrough.
        if isinstance(route, Mount):
            continue
        raise TypeError(
            f"iter_routes does not recognize {route!r} ({type(route).__name__}). "
            "It walks the app's real route surface for the guards that prove a "
            "property of EVERY route, so an unknown node is refused rather than "
            "skipped - a skip would shrink that surface silently, which is the "
            "exact regression this function was written to close."
        )


def iter_api_routes(target: Any) -> Iterator[APIRoute]:
    """The ``APIRoute``s of ``iter_routes``: the documented API surface."""
    return (route for route in iter_routes(target) if isinstance(route, APIRoute))
