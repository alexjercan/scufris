"""Running an approved command as root, with output and a way to stop it.

The inspection package runs commands with ``subprocess.run`` and waits for the
answer (``scufris_host``'s ``run.py``). A privileged action needs two things that
does not give: the operator watches the output as it happens, and they can stop
it. So this is the one place in ``scufris.hostd`` that spawns a process, and it
is deliberately the only one.

Cancellation kills the whole PROCESS GROUP, not the child. ``nixos-rebuild``
and ``nix-collect-garbage`` are drivers that spawn their own children; killing
only the parent leaves root-owned work running with nothing watching it and no
record of when it stopped.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import time
from typing import Callable, Protocol, runtime_checkable

from scufris_host import CommandResult, Outcome

logger = logging.getLogger(__name__)

# How long a killed process group is given to die politely before SIGKILL.
KILL_GRACE_SECONDS = 5.0

# Per-stream cap on the output kept in the RESULT record. The operator sees the
# whole stream live; the record keeps a bounded tail so one chatty command
# cannot dominate the audit log.
MAX_KEPT_OUTPUT = 8000

# A sink for live output: (stream name, text). Never awaited, so a slow consumer
# cannot stall the command it is watching.
OutputSink = Callable[[str, str], None]


@runtime_checkable
class Executor(Protocol):
    """The seam an approved action runs through. Tests provide a fake."""

    async def __call__(
        self,
        argv: list[str],
        *,
        timeout: float,
        on_output: OutputSink,
    ) -> CommandResult:
        """Run ``argv``, streaming output to ``on_output``. Never raises except
        ``asyncio.CancelledError``, which means the caller stopped it."""
        ...


async def _pump(
    stream: asyncio.StreamReader | None, name: str, on_output: OutputSink
) -> str:
    """Forward one pipe to the sink, returning the tail that is kept."""
    if stream is None:
        return ""
    kept: list[str] = []
    kept_len = 0
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode("utf-8", "replace")
        on_output(name, text)
        kept.append(text)
        kept_len += len(text)
        while kept_len > MAX_KEPT_OUTPUT and len(kept) > 1:
            kept_len -= len(kept.pop(0))
    return "".join(kept)[-MAX_KEPT_OUTPUT:]


async def run_action(
    argv: list[str],
    *,
    timeout: float,
    on_output: OutputSink,
) -> CommandResult:
    """Execute ``argv`` as this process's user, streaming its output.

    ``shell=False`` with an explicit argv, and the argv itself was built by
    ``actions.build_plan`` - nothing here reinterprets it.
    """
    if not argv:
        return CommandResult(argv=[], outcome=Outcome.FAILED, stderr="empty command")
    exe = shutil.which(argv[0])
    if exe is None:
        return CommandResult(argv=argv, outcome=Outcome.MISSING)
    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            exe,
            *argv[1:],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Its own session, so the whole tree can be signalled at once and a
            # signal sent to us is not inherited by root-owned work.
            start_new_session=True,
        )
    except OSError as exc:
        return CommandResult(argv=argv, outcome=Outcome.FAILED, stderr=str(exc))

    pumps = asyncio.gather(
        _pump(proc.stdout, "stdout", on_output),
        _pump(proc.stderr, "stderr", on_output),
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            _await_completion(proc, pumps), timeout=timeout
        )
    except asyncio.TimeoutError:
        await _terminate(proc)
        pumps.cancel()
        return CommandResult(
            argv=argv,
            outcome=Outcome.TIMEOUT,
            duration_seconds=time.monotonic() - started,
        )
    except asyncio.CancelledError:
        # The operator stopped it. Kill the group, then let the cancellation
        # propagate so the caller records a CANCELLED outcome rather than a
        # failure it has to guess at.
        await _terminate(proc)
        pumps.cancel()
        raise
    elapsed = time.monotonic() - started
    code = proc.returncode
    outcome = Outcome.OK if code == 0 else Outcome.FAILED
    return CommandResult(
        argv=argv,
        outcome=outcome,
        stdout=stdout,
        stderr=stderr,
        returncode=code,
        duration_seconds=elapsed,
    )


async def _await_completion(
    proc: asyncio.subprocess.Process, pumps: "asyncio.Future[tuple[str, str]]"
) -> tuple[str, str]:
    collected = await pumps
    await proc.wait()
    return collected[0], collected[1]


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    """Signal the child's whole process group, escalating if it does not die."""
    if proc.returncode is not None:
        return
    try:
        group = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        group = None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            if group is not None:
                os.killpg(group, sig)
            else:
                proc.send_signal(sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=KILL_GRACE_SECONDS)
            return
        except asyncio.TimeoutError:
            logger.warning("hostd: pid %s survived %s, escalating", proc.pid, sig.name)
    return


class FakeExecutor:
    """A scripted ``Executor`` for tests and the example script.

    Emits its canned output, optionally pausing so a test can cancel mid-apply,
    and records what it was asked to run. Nothing here spawns a process, so the
    whole framework - including its cancellation path - is exercisable off a
    real host.
    """

    def __init__(
        self,
        *,
        output: list[tuple[str, str]] | None = None,
        result: CommandResult | None = None,
        hang: bool = False,
    ) -> None:
        self.output = output or []
        self.result = result
        self.hang = hang
        self.calls: list[list[str]] = []
        # Set once the fake has emitted its output and is about to block, so a
        # test can wait for the apply to be genuinely in flight before cancelling.
        self.started = asyncio.Event()

    async def __call__(
        self,
        argv: list[str],
        *,
        timeout: float,
        on_output: OutputSink,
    ) -> CommandResult:
        self.calls.append(list(argv))
        for stream, text in self.output:
            on_output(stream, text)
        self.started.set()
        if self.hang:
            await asyncio.sleep(3600)
        if self.result is not None:
            return self.result.model_copy(update={"argv": list(argv)})
        return CommandResult(
            argv=list(argv), outcome=Outcome.OK, returncode=0, duration_seconds=0.01
        )
