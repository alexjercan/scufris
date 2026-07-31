"""The registry of configuration changes, and the build that fills it.

The build runs as the OPERATOR, never as root. Nix evaluation reads files with
the evaluating user's privileges, so a configuration evaluated as root could
read a host key or a sops age key into a derivation output; as the operator that
read simply fails.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

from ..host.run import Outcome, Runner, run_command
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

# Bounded like the action registry: config changes are short-lived records and
# the audit log is the durable half.
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
    """The app's bounded registry of configuration changes."""

    def __init__(
        self,
        *,
        max_changes: int = MAX_CHANGES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._changes: "OrderedDict[str, ConfigChange]" = OrderedDict()
        self._max = max_changes
        self._now = clock

    def put(self, change: ConfigChange) -> ConfigChange:
        if not change.created_at:
            change.created_at = self._now()
        self._changes[change.id] = change
        self._reap()
        return change

    def get(self, change_id: str) -> ConfigChange:
        try:
            return self._changes[change_id]
        except KeyError as exc:
            raise UnknownChange(change_id) from exc

    def list(self) -> list[ConfigChange]:
        return list(reversed(self._changes.values()))

    def building_for(self, repo: str) -> ConfigChange | None:
        return next(
            (
                change
                for change in self._changes.values()
                if change.state is ChangeState.BUILDING and change.repo == repo
            ),
            None,
        )

    def _reap(self) -> None:
        while len(self._changes) > self._max:
            for key, change in self._changes.items():
                if change.state is not ChangeState.BUILDING:
                    del self._changes[key]
                    break
            else:
                self._changes.popitem(last=False)


# Called with the built toplevel; returns the host-action id that now carries it.
Propose = Callable[[ConfigChange], Awaitable[str]]


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
        self, change: ConfigChange, propose: Propose
    ) -> AsyncIterator[ConfigBuildEvent]:
        """Build ``change``, then propose the activation of what was built.

        A build failure is TERMINAL here: the record keeps the log tail and no
        proposal is ever created, so a configuration that does not build has no
        route to activation at all - not because a check refuses it, but because
        the thing an approval would act on does not exist.
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
            raise

        if result.outcome is not Outcome.OK:
            change.state = ChangeState.FAILED
            change.log_tail = "".join(collected)[-MAX_LOG_TAIL:]
            change.error = f"the configuration did not build: {result.reason()}"
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
            yield ConfigBuildError(detail=change.error)
            return

        change.toplevel = toplevel
        try:
            change.action_id = await propose(change)
        except Exception as exc:  # noqa: BLE001 - the refusal is the answer
            change.state = ChangeState.FAILED
            change.error = f"the built configuration was not accepted: {exc}"
            yield ConfigBuildError(detail=change.error)
            return
        change.state = ChangeState.PROPOSED
        yield ConfigBuildDone(change=change)
