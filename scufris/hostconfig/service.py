"""The configuration-change flow, as one object the HTTP layer delegates to.

The route used to hold all of it: resolve the ref, refuse a second build of the
same repository, mint the record, start the supervised build, wire the propose
and save callbacks. None of that is an HTTP concern - it is what a configuration
change IS - so it lives here and the router does what a router should, which is
translate this module's refusals into statuses.

The activation proposal is injected rather than built here. This package knows
how to turn a ref into a store path; who is allowed to be told about the result,
and how a proposal reaches the operator, belongs to the approval service.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

from ..config import Settings
from ..eventbus import EventBus
from ..hostd.audit import Requester
from .changes import (
    ChangeInFlight,
    ConfigChangeBuilder,
    ConfigChangeStore,
    UnknownChange,
)
from .models import (
    ChangeState,
    ConfigBuildEvent,
    ConfigChange,
    ConfigSupervisor,
)
from .resolve import default_attr

#: Propose activating what was built, and return the proposal id. Injected so
#: this package does not have to know about the approval queue.
Proposer = Callable[[ConfigChange, Requester], Awaitable[str]]


class NoRunningBuild(RuntimeError):
    """A cancel was asked for on a change with nothing building."""


class ConfigChangeService:
    """Resolve, build and cancel configuration changes.

    Holds the registry, the builder, the build supervisor, the proposer and the
    settings that supply the defaults - so a caller supplies a ref and a
    requester and nothing else.
    """

    def __init__(
        self,
        *,
        store: ConfigChangeStore,
        builder: ConfigChangeBuilder,
        supervisor: ConfigSupervisor,
        propose: Proposer,
        settings: Settings,
    ) -> None:
        self._store = store
        self._builder = builder
        self._supervisor = supervisor
        self._propose = propose
        self._settings = settings

    async def list(self) -> list[ConfigChange]:
        """Changes, newest first. Offloaded, like every store call from the loop."""
        return await asyncio.to_thread(self._store.list)

    async def get(self, change_id: str) -> ConfigChange:
        """One change. Raises ``UnknownChange``."""
        return await asyncio.to_thread(self._store.get, change_id)

    def bus(self, change: ConfigChange) -> EventBus[ConfigBuildEvent] | None:
        """The live build's event bus, or None when there is nothing to stream."""
        return self._supervisor.bus(change.run_id) if change.run_id else None

    async def start(
        self,
        requester: Requester,
        *,
        ref: str = "",
        repo: str = "",
        attr: str = "",
    ) -> ConfigChange:
        """Resolve a ref and start building it, proposing the activation after.

        Raises ``ConfigChangeRefused`` when the ref or the repository is not
        acceptable, and ``ChangeInFlight`` when this repository already has a
        build running. The build runs as this process's user - never root - and
        reads the tree from the COMMIT, so this cannot touch the repository's
        working tree.
        """
        target = Path(repo).expanduser() if repo else self._settings.host_config_repo
        attribute = attr or self._settings.host_config_attr or default_attr()
        # git reads only: milliseconds, so the caller answers immediately. The
        # flake evaluation and the build both happen in the run.
        _main, resolved = await asyncio.to_thread(
            self._builder.resolve,
            target,
            ref or "HEAD",
            allowed=self._settings.host_config_repo,
        )
        in_flight = await asyncio.to_thread(self._store.building_for, resolved.repo)
        if in_flight is not None:
            raise ChangeInFlight(
                f"a configuration build is already running for {resolved.repo} "
                f"({in_flight.id}, {in_flight.resolved.ref}). Wait for it or "
                "cancel it before starting another."
            )
        # The run id is on the record BEFORE it is stored: a row written and then
        # mutated would be a second write, and a restart between the two would
        # leave a change nothing could stream.
        change_id = uuid.uuid4().hex
        change = await asyncio.to_thread(
            self._store.put,
            ConfigChange(
                id=change_id,
                resolved=resolved,
                attr=attribute,
                run_id=f"config:{change_id}",
                agent=requester.agent,
                requested_by=requester.actor,
            ),
        )

        async def _propose(built: ConfigChange) -> str:
            return await self._propose(built, requester)

        async def _save(built: ConfigChange) -> None:
            """Write the build's own state transitions back to the registry.

            The builder holds no store (20260803-002141 DECISION.md 1) and runs
            on the loop thread, so the write is offloaded here rather than there.
            """
            await asyncio.to_thread(self._store.put, built)

        def _stream() -> AsyncIterator[ConfigBuildEvent]:
            return self._builder.stream(change, _propose, _save)

        self._supervisor.start(
            change.run_id,
            _stream,
            # One build at a time per repository. The refusal above is the
            # visible half; this is what makes it true even in a race.
            serialize_key=f"config:{resolved.repo}",
        )
        return change

    async def cancel(self, change_id: str) -> ConfigChange:
        """Stop a build that is running.

        Raises ``UnknownChange`` when there is no such change and
        ``NoRunningBuild`` when it has nothing in flight. Not operator-only:
        a build holds no privilege and stopping it undoes nothing. What IS
        operator-only is approving the activation it produces.
        """
        change = await self.get(change_id)
        if change.state is not ChangeState.BUILDING or not change.run_id:
            raise NoRunningBuild("this change has no running build to cancel")
        if not self._supervisor.cancel(change.run_id):
            raise NoRunningBuild("this change has no running build to cancel")
        return change


__all__ = [
    "ChangeInFlight",
    "ConfigChangeService",
    "NoRunningBuild",
    "Proposer",
    "UnknownChange",
]
