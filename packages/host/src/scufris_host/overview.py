"""The cached host overview: one slot, a TTL floor, and single-flight collection.

Its own module rather than a piece of the application factory because the whole
thing is one policy - how often the subprocess-backed overview may actually be
collected - and a route should be able to ask for the overview in one call
instead of reassembling that policy out of a cache object, a floor constant and
a lock.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from .inspector import HostInspector, HostOverview

# Floor on the overview cache's TTL. The endpoint is subprocess-backed, so
# "uncached" is never a sensible configuration - a 0 in the env would otherwise
# turn every poll of every open tab into its own systemctl + nixos-rebuild
# fan-out, which is exactly what the cache exists to prevent.
MIN_HOST_OVERVIEW_TTL = 2.0


class HostOverviewCache:
    """One slot holding the most recent host overview, with a TTL.

    The overview costs several subprocesses. Without this, every open dashboard
    tab (and every poll of every tab) would run its own `nixos-rebuild
    list-generations`. One slot, not a keyed dict: there is exactly one host, so
    there is nothing to key on and nothing to reap - the bounded-registry problem
    cannot arise here.
    """

    def __init__(
        self,
        inspector: HostInspector,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inspector = inspector
        self._ttl = max(MIN_HOST_OVERVIEW_TTL, ttl_seconds)
        # Injected so a test can assert the TTL BOUNDARY without sleeping on a
        # wall clock or monkeypatching the stdlib time module globally.
        self._clock = clock
        self._value: HostOverview | None = None
        self._collected_at = 0.0
        # Single-flight guard for the collection. An asyncio lock, not a
        # threading one: a waiting request suspends on the loop rather than
        # holding a default-executor thread, so a slow nixos-rebuild cannot
        # starve the executor that the rest of the app shares.
        self._lock = asyncio.Lock()

    def _fresh(self) -> HostOverview | None:
        """The cached value if it is still within the TTL, else None."""
        cached = self._value
        if cached is None:
            return None
        return cached if self._clock() - self._collected_at < self._ttl else None

    def get(self) -> HostOverview:
        """The cached overview, collecting it when the TTL has expired.

        SYNCHRONOUS and not internally locked - ``overview`` is what a route
        calls. Kept separate because the collection is blocking subprocess work:
        it is what gets handed to ``asyncio.to_thread``, and a ``threading.Lock``
        around it would be held across the whole collection while every
        concurrent caller occupied one of the shared default-executor threads,
        so a slow `nixos-rebuild` would starve the executor the rest of the app
        uses.
        """
        cached = self._fresh()
        if cached is not None:
            return cached
        collected = self._inspector.overview()
        self._value = collected
        self._collected_at = self._clock()
        return collected

    async def overview(self) -> HostOverview:
        """The overview, collected off the event loop and at most once at a time.

        Serving a fresh value takes no lock at all: the common case is a cache
        hit, and queueing those behind a collection in flight would make every
        polling tab wait on it.
        """
        cached = self._fresh()
        if cached is not None:
            return cached
        async with self._lock:
            # Re-check inside the lock: a burst that queued here while the first
            # request collected must serve that result, not collect again.
            return await asyncio.to_thread(self.get)
