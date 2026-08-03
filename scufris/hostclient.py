"""The app's side of the hostd socket.

The app never runs a privileged command itself; it asks the helper, which owns
the verbs, the proposals and the audit log. So this module is small on purpose:
connect, send one authenticated request, read frames.

Two things it is careful about:

- **The secret arrives THROUGH ``os.environ``, and is stripped from every child
  process.** A sops secret reaches the unit as an ``EnvironmentFile``, so unlike
  ``auth_api_token`` (minted in-process) this one is present in the environment
  by construction and "we never put it there" is not available as a defence.
  What protects it is ``config.SECRET_ENV_VARS``, removed from every agent
  subprocess environment by ``agent.agent_subprocess_env`` - otherwise the agent
  CLI, and hence every shell command the model runs, would hold the credential
  for the socket and could apply host actions with no operator at all. This
  paragraph claimed the opposite until review round 2, R2.3.
- **Apply is a stream that can be cut.** Closing the generator sends a cancel
  frame and drops the connection; the helper kills the process group and records
  the cancellation. There is no "fire and forget" path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import AsyncIterator, Callable, Literal

from pydantic import BaseModel

from scufris_hostd import (
    ActionKind,
    AuditRecord,
    ErrorCode,
    ProposalView,
    Request,
    Requester,
    ResultFrame,
    Verb,
    encode,
)

from .eventbus import EventBus
from .supervisor import Supervisor

logger = logging.getLogger(__name__)

# A frame is a single line. The audit tail is the largest answer and is bounded
# by the protocol's own limit.
READ_LIMIT = 8 * 1024 * 1024

# Connecting to a local unix socket either works immediately or the helper is
# not there.
CONNECT_TIMEOUT = 5.0


class HostdUnavailable(Exception):
    """The helper is not reachable: not enabled, not running, or not permitted.

    Distinct from a refusal on purpose. "The privileged helper is not
    configured" is an operator-facing deployment fact; "the helper said no" is
    an answer about the action.
    """


class HostdError(Exception):
    """The helper refused, with its own code."""

    def __init__(self, code: ErrorCode, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


# --- what an apply publishes on the bus ---------------------------------
#
# Host apply events are their OWN type, not members of the agent's StreamEvent
# union: a root command's output must not be renderable as model text by a chat
# surface that happens to receive it.


class HostApplyOutput(BaseModel):
    type: Literal["output"] = "output"
    stream: str
    text: str


class HostApplyResult(BaseModel):
    type: Literal["result"] = "result"
    result: ResultFrame


class HostApplyError(BaseModel):
    type: Literal["error"] = "error"
    detail: str
    code: ErrorCode = ErrorCode.INTERNAL


HostApplyEvent = HostApplyOutput | HostApplyResult | HostApplyError

HostSupervisor = Supervisor[HostApplyEvent]
HostApplyBus = EventBus[HostApplyEvent]


def _host_error_event(detail: str) -> HostApplyEvent:
    return HostApplyError(detail=detail)


def _host_error_detail(event: HostApplyEvent) -> str | None:
    return event.detail if isinstance(event, HostApplyError) else None


def host_supervisor(
    *,
    max_concurrent: int = 1,
    max_history: int = 200,
    clock: Callable[[], float] = time.time,
) -> HostSupervisor:
    """A supervisor for host applies.

    ``max_concurrent=1`` by default, and that is a decision rather than a
    default: two root commands running at once on one machine is not something
    an operator approved, and the serialize key is per-action rather than
    global.
    """
    return Supervisor(
        error_event=_host_error_event,
        error_detail=_host_error_detail,
        max_concurrent=max_concurrent,
        max_history=max_history,
        clock=clock,
    )


class HostdClient:
    """Talks to a running ``scufris-hostd`` over its unix socket."""

    def __init__(self, socket_path: Path, secret: str) -> None:
        self._path = Path(socket_path)
        self._secret = secret

    @property
    def configured(self) -> bool:
        """Whether the app has both halves it needs to even try."""
        return bool(self._secret) and bool(self._path)

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if not self.configured:
            raise HostdUnavailable(
                "the privileged host helper is not configured: set "
                "SCUFRIS_HOSTD_SOCKET and SCUFRIS_HOSTD_SECRET, and enable "
                "services.scufris-hostd in the NixOS configuration"
            )
        try:
            return await asyncio.wait_for(
                asyncio.open_unix_connection(str(self._path), limit=READ_LIMIT),
                timeout=CONNECT_TIMEOUT,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            raise HostdUnavailable(
                f"the privileged host helper is not answering on {self._path}: {exc}"
            ) from exc

    def _request(self, verb: Verb, **fields: object) -> Request:
        payload: dict[str, object] = {
            "verb": verb,
            "secret": self._secret,
            **fields,
        }
        return Request.model_validate(payload)

    async def _one_shot(self, request: Request) -> dict[str, object]:
        reader, writer = await self._connect()
        try:
            writer.write(encode(request))
            await writer.drain()
            frame = await self._read_frame(reader)
        finally:
            writer.close()
            with suppress(ConnectionResetError, BrokenPipeError, OSError):
                await writer.wait_closed()
        _raise_for_error(frame)
        return frame

    async def _read_frame(self, reader: asyncio.StreamReader) -> dict[str, object]:
        try:
            line = await reader.readuntil(b"\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError) as exc:
            raise HostdUnavailable(
                "the privileged host helper closed the connection without an answer"
            ) from exc
        try:
            parsed = json.loads(line)
        except ValueError as exc:
            raise HostdUnavailable(
                "the privileged host helper sent something this build cannot read"
            ) from exc
        if not isinstance(parsed, dict):
            raise HostdUnavailable("the privileged host helper sent a non-frame")
        return parsed

    # --- verbs -----------------------------------------------------------

    async def propose(
        self,
        kind: ActionKind,
        args: dict[str, object],
        requester: Requester,
    ) -> ProposalView:
        frame = await self._one_shot(
            self._request(
                Verb.PROPOSE,
                kind=kind,
                args=args,
                requester=requester.model_dump(),
            )
        )
        return ProposalView.model_validate(frame["proposal"])

    async def deny(
        self, proposal_id: str, *, operator: str, reason: str = ""
    ) -> ProposalView:
        frame = await self._one_shot(
            self._request(
                Verb.DENY,
                proposal_id=proposal_id,
                approved_by=operator,
                reason=reason,
            )
        )
        return ProposalView.model_validate(frame["proposal"])

    async def list_pending(self) -> list[ProposalView]:
        """The proposals the helper still holds PENDING, oldest first.

        Read-only, and the app's way back to a truthful queue: its own registry is
        in-memory by design (the helper owns proposals), so after a restart this is
        what tells it which approvals are still live rather than a second persisted
        copy of the helper's state.
        """
        frame = await self._one_shot(self._request(Verb.LIST_PENDING))
        rows = frame.get("proposals", [])
        if not isinstance(rows, list):
            return []
        return [ProposalView.model_validate(row) for row in rows]

    async def audit_tail(self, limit: int = 50) -> list[AuditRecord]:
        frame = await self._one_shot(self._request(Verb.AUDIT_TAIL, limit=limit))
        rows = frame.get("records", [])
        if not isinstance(rows, list):
            return []
        return [AuditRecord.model_validate(row) for row in rows]

    async def apply(
        self, proposal_id: str, *, approved_by: str
    ) -> AsyncIterator[HostApplyEvent]:
        """Stream one approved action's output and its terminal result.

        Cancelling the consumer (``supervisor.cancel``) closes this generator,
        which sends a cancel frame and drops the connection. Either one tells
        the helper to signal the process group; it records the cancellation
        whichever arrives first.
        """
        reader, writer = await self._connect()
        request = self._request(
            Verb.APPLY, proposal_id=proposal_id, approved_by=approved_by
        )
        finished = False
        try:
            writer.write(encode(request))
            await writer.drain()
            while True:
                frame = await self._read_frame(reader)
                kind = frame.get("type")
                if kind == "output":
                    yield HostApplyOutput.model_validate(frame)
                    continue
                if kind == "result":
                    finished = True
                    yield HostApplyResult(result=ResultFrame.model_validate(frame))
                    return
                if kind == "error":
                    finished = True
                    yield HostApplyError(
                        detail=str(frame.get("detail", "")),
                        code=ErrorCode(str(frame.get("code", ErrorCode.INTERNAL))),
                    )
                    return
                logger.info("hostd: ignoring an unknown frame type %r", kind)
        finally:
            # Unfinished means the consumer stopped early - a supervisor cancel
            # closes this generator, which arrives as GeneratorExit at the
            # yield, NOT as CancelledError. So the cancel frame is written here
            # rather than in an except clause, and written SYNCHRONOUSLY: an
            # async generator being closed must not await a drain.
            if not finished:
                with suppress(ConnectionResetError, BrokenPipeError, OSError):
                    writer.write(
                        encode(self._request(Verb.CANCEL, proposal_id=proposal_id))
                    )
            writer.close()


def _raise_for_error(frame: dict[str, object]) -> None:
    if frame.get("type") == "error":
        raise HostdError(
            ErrorCode(str(frame.get("code", ErrorCode.INTERNAL))),
            str(frame.get("detail", "")),
        )
