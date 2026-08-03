"""The registry of configuration changes, and the build that fills it.

The build runs as the OPERATOR, never as root. Nix evaluation reads files with
the evaluating user's privileges, so a configuration evaluated as root could
read a host key or a sops age key into a derivation output; as the operator that
read simply fails.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from sqlalchemy import Connection, Row, func, insert, select
from sqlalchemy import delete as sql_delete
from sqlalchemy import update as sql_update

from scufris_host import Outcome, Runner, run_command

from ..db import Database
from ..db.models import ConfigChangeRow
from ..hostd.executor import Executor, run_action
from .models import (
    ChangeState,
    ConfigBuildDone,
    ConfigBuildError,
    ConfigBuildEvent,
    ConfigBuildOutput,
    ConfigChange,
    ConfigChangeRefused,
    Resolved,
)
from .resolve import build_argv, check_attr, flake_url, resolve, toplevel_from

# Bounded like the action registry: config changes are short-lived build records,
# and what an activation did to the host is the audit log's, not this table's.
# Durability makes the bound matter more, not less - the rows no longer go away
# on their own.
MAX_CHANGES = 100

# How much of a failed build's log is kept on the record. A nix build failure is
# chatty and the useful part is the end.
MAX_LOG_TAIL = 16000


class UnknownChange(KeyError):
    """No such configuration change id."""


class ChangeInFlight(RuntimeError):
    """A build is already running against this repository.

    Refused rather than queued, and the difference matters to the person
    waiting: a queued NixOS build can sit for an hour behind another one with no
    sign of why. Two builds of the same repository also contend for the same
    evaluation and the same store, so serializing them buys nothing but
    confusion.
    """


class ConfigChangeStore:
    """The app's bounded registry of configuration changes.

    Every method is ONE unit of work on the app's one transactional boundary, and
    every one is SYNCHRONOUS: ``Database.transaction()`` refuses a thread with a
    running event loop, so the ``async def`` routes in ``app.py`` and the
    builder's ``save`` callback all offload with ``asyncio.to_thread``.

    :meth:`put` is an UPSERT because the writer that matters here is not the
    request: a build writes back over minutes to hours as it moves through its
    states, and re-storing a change it already stored is the ordinary case rather
    than a special one (20260803-002141 DECISION.md 1).
    """

    def __init__(
        self,
        db: Database,
        *,
        max_changes: int = MAX_CHANGES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._db = db
        self._max = max_changes
        self._now = clock

    def put(self, change: ConfigChange) -> ConfigChange:
        """Record a change, or write back one whose id is already known."""
        if not change.created_at:
            change.created_at = self._now()
        with self._db.transaction() as conn:
            values = _values(change)
            if _row(conn, change.id) is not None:
                # `seq` is not in `values`, so the list position a change was
                # given when it was created survives every write-back the build
                # makes to it.
                conn.execute(
                    sql_update(ConfigChangeRow)
                    .where(ConfigChangeRow.id == change.id)
                    .values(**values)
                )
                return change
            # `seq` is assigned here, inside the inserting transaction: the begin
            # is immediate, so two changes arriving together cannot claim one
            # number. See `ConfigChangeRow` for why it is not the rowid.
            nxt = conn.execute(select(func.coalesce(func.max(ConfigChangeRow.seq), 0)))
            conn.execute(
                insert(ConfigChangeRow).values(seq=nxt.scalar_one() + 1, **values)
            )
            self._reap(conn)
        return change

    def get(self, change_id: str) -> ConfigChange:
        with self._db.transaction() as conn:
            row = _row(conn, change_id)
            if row is None:
                raise UnknownChange(change_id)
            return _change(row)

    def list(self) -> list[ConfigChange]:
        """Newest first - a list of builds is read from the top."""
        with self._db.transaction() as conn:
            rows = conn.execute(
                select(ConfigChangeRow.__table__).order_by(ConfigChangeRow.seq.desc())
            ).all()
        return [_change(row) for row in rows]

    def building_for(self, repo: str) -> ConfigChange | None:
        """The change currently building ``repo``, if there is one.

        The repository is matched in PYTHON, on the handful of rows that are
        building, rather than on a column that would duplicate a field of the
        `resolved` JSON.
        """
        with self._db.transaction() as conn:
            rows = conn.execute(
                select(ConfigChangeRow.__table__)
                .where(ConfigChangeRow.state == str(ChangeState.BUILDING))
                .order_by(ConfigChangeRow.seq.asc())
            ).all()
        return next(
            (change for row in rows if (change := _change(row)).repo == repo), None
        )

    def abandon_builds(self) -> None:
        """Fail every change left BUILDING by a restart.

        Run once at startup. In memory a crashed build vanished with the process;
        on a row it would stay `building` forever, :meth:`building_for` would keep
        answering it, and every later build of that repository would be refused
        with a 409 that cancelling cannot clear - cancelling needs a live
        supervisor run, and the supervisor was restarted too. This is honest as
        well as convenient: the build is in fact not running, nothing was
        proposed, and FAILED is the terminal state that carries no proposal.
        20260803-002141 DECISION.md 2.
        """
        with self._db.transaction() as conn:
            conn.execute(
                sql_update(ConfigChangeRow)
                .where(ConfigChangeRow.state == str(ChangeState.BUILDING))
                .values(
                    state=str(ChangeState.FAILED),
                    error=(
                        "the server restarted while this configuration was "
                        "building, so the build was lost and nothing was "
                        "proposed. Start it again - nix keeps what it already "
                        "built, so the next attempt reuses it."
                    ),
                )
            )

    def _reap(self, conn: Connection, /) -> None:
        """Hold the bound, on an OPEN connection, non-building changes first.

        A building change is never dropped ahead of a settled one: it is the one
        with a live run behind it. When everything is building the oldest goes
        anyway rather than growing without bound.
        """
        over = conn.execute(
            select(func.count()).select_from(ConfigChangeRow)
        ).scalar_one()
        over -= self._max
        if over <= 0:
            return
        doomed = (
            select(ConfigChangeRow.id)
            .order_by(
                (ConfigChangeRow.state == str(ChangeState.BUILDING)).asc(),
                ConfigChangeRow.seq.asc(),
            )
            .limit(over)
        )
        conn.execute(sql_delete(ConfigChangeRow).where(ConfigChangeRow.id.in_(doomed)))


def _row(conn: Connection, change_id: str) -> Row[Any] | None:
    return conn.execute(
        select(ConfigChangeRow.__table__).where(ConfigChangeRow.id == change_id)
    ).first()


def _values(change: ConfigChange) -> dict[str, Any]:
    """One change as columns. ``resolved`` stays JSON text - it is a nested model
    and nothing queries inside it."""
    return {
        "id": change.id,
        "resolved": change.resolved.model_dump_json(),
        "attr": change.attr,
        "state": str(change.state),
        "toplevel": change.toplevel,
        "action_id": change.action_id,
        "run_id": change.run_id,
        "log_tail": change.log_tail,
        "error": change.error,
        "created_at": change.created_at,
        "agent": change.agent,
        "requested_by": change.requested_by,
    }


def _change(row: Row[Any], /) -> ConfigChange:
    """The pydantic record for one selected row. Nothing else leaves the store."""
    fields = dict(row._mapping)
    fields.pop("seq", None)
    fields["resolved"] = json.loads(fields["resolved"])
    return ConfigChange.model_validate(fields)


# Called with the built toplevel; returns the host-action id that now carries it.
Propose = Callable[[ConfigChange], Awaitable[str]]

# Called after every state transition, to write the change back to wherever it
# lives. The builder is deliberately store-free (20260803-002141 DECISION.md 1),
# and this is AWAITED because the store's transaction cannot be opened on the
# loop thread: `app.py` passes a wrapper that offloads it.
Save = Callable[[ConfigChange], Awaitable[None]]


class ConfigChangeBuilder:
    """Builds a resolved change and hands the result to the propose step."""

    def __init__(
        self,
        *,
        runner: Runner = run_command,
        executor: Executor = run_action,
        build_timeout: float = 7200.0,
    ) -> None:
        self._runner = runner
        self._executor = executor
        self._timeout = build_timeout

    # The git and evaluation reads go through the same seam as the build, so a
    # test drives the whole pipeline through one injection rather than three.

    def resolve(
        self, repo: Path, ref: str, *, allowed: Path | None = None
    ) -> tuple[Path, Resolved]:
        return resolve(repo, ref, runner=self._runner, allowed=allowed)

    def check_attr(self, url: str, attr: str) -> None:
        check_attr(url, attr, runner=self._runner)

    async def stream(
        self, change: ConfigChange, propose: Propose, save: Save
    ) -> AsyncIterator[ConfigBuildEvent]:
        """Build ``change``, then propose the activation of what was built.

        A build failure is TERMINAL here: the record keeps the log tail and no
        proposal is ever created, so a configuration that does not build has no
        route to activation at all - not because a check refuses it, but because
        the thing an approval would act on does not exist.

        ``save`` is called after EVERY state transition. This builder mutates the
        ``ConfigChange`` it was given and nothing else; a registry that hands out
        rows rather than objects would otherwise see none of it, and the polls in
        ``tests/test_nixos_config_change.py`` would spin until they time out.
        """
        url = flake_url(Path(change.resolved.repo), change.resolved)
        # The probe is a full flake EVALUATION - measured 6.4s warm on this host
        # and slower cold - so it runs HERE rather than in the request that
        # started this. The MCP tool's own API timeout is 15s, so probing before
        # returning made the one call an agent always makes on a changed
        # configuration the one most likely to report a timeout for a build that
        # was in fact running.
        try:
            self.check_attr(url, change.attr)
        except ConfigChangeRefused as exc:
            change.state = ChangeState.FAILED
            change.error = str(exc)
            await save(change)
            yield ConfigBuildError(detail=change.error)
            return
        argv = build_argv(url, change.attr)
        collected: list[str] = []
        queue: "asyncio.Queue[ConfigBuildEvent]" = asyncio.Queue()

        def sink(stream: str, text: str) -> None:
            collected.append(text)
            # Bounded: a nix build log can run to megabytes, and the operator is
            # watching the live stream anyway.
            if len(collected) > 4000:
                del collected[: len(collected) - 4000]
            queue.put_nowait(ConfigBuildOutput(stream=stream, text=text))

        yield ConfigBuildOutput(stream="stdout", text=f"$ {' '.join(argv)}\n")
        # The executor's sink is synchronous, so the log reaches the consumer
        # through a queue while the build runs as its own task. Yielding the
        # collected output after the fact would turn a live build log into a
        # transcript that arrives when it is no longer useful.
        build = asyncio.ensure_future(
            self._executor(argv, timeout=self._timeout, on_output=sink)
        )
        try:
            while True:
                pending = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait(
                    {pending, build}, return_when=asyncio.FIRST_COMPLETED
                )
                if pending in done:
                    yield pending.result()
                    continue
                pending.cancel()
                break
            while not queue.empty():
                yield queue.get_nowait()
            result = await build
        except (asyncio.CancelledError, GeneratorExit):
            # The operator stopped watching, or cancelled outright. Kill the
            # build; nix keeps whatever it finished in the store, so the next
            # attempt resumes rather than starting over.
            build.cancel()
            change.state = ChangeState.CANCELLED
            change.error = (
                "the build was stopped before it finished, so nothing was built "
                "and nothing was proposed. What nix had already built stays in "
                "the store and the next attempt reuses it."
            )
            await save(change)
            raise

        if result.outcome is not Outcome.OK:
            change.state = ChangeState.FAILED
            change.log_tail = "".join(collected)[-MAX_LOG_TAIL:]
            change.error = f"the configuration did not build: {result.reason()}"
            await save(change)
            yield ConfigBuildError(detail=change.error)
            return

        toplevel = toplevel_from(result.stdout)
        if not toplevel:
            change.state = ChangeState.FAILED
            change.log_tail = "".join(collected)[-MAX_LOG_TAIL:]
            change.error = (
                "the build reported success but printed no store path, so there "
                "is nothing identifiable to activate"
            )
            await save(change)
            yield ConfigBuildError(detail=change.error)
            return

        change.toplevel = toplevel
        try:
            change.action_id = await propose(change)
        except Exception as exc:  # noqa: BLE001 - the refusal is the answer
            change.state = ChangeState.FAILED
            change.error = f"the built configuration was not accepted: {exc}"
            await save(change)
            yield ConfigBuildError(detail=change.error)
            return
        change.state = ChangeState.PROPOSED
        await save(change)
        yield ConfigBuildDone(change=change)
