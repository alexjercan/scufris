"""The unix socket in front of the engine.

One request per connection, authenticated per frame. The socket is the only way
into the privileged process, so everything it refuses - an unknown verb, a bad
secret, an oversized line - is refused before any code that builds a command
runs.

Cancellation arrives two ways, and both mean the same thing: a ``cancel`` frame
on the live connection, or the connection simply dropping. A root command must
not keep running for a caller that is no longer there, so either one kills the
process group and lands a ``cancelled`` record.
"""

from __future__ import annotations

import asyncio
import grp
import hmac
import logging
import os
import stat
from pathlib import Path

from pydantic import BaseModel, ValidationError

from .engine import HostdEngine, HostdRefusal
from .protocol import (
    ErrorCode,
    ErrorFrame,
    OutputFrame,
    ProposalFrame,
    Request,
    Verb,
    encode,
)

logger = logging.getLogger(__name__)

# A request frame is a small JSON object. Anything larger is not a request this
# helper has a verb for.
MAX_REQUEST_BYTES = 64 * 1024

# How much streamed output one apply may forward before it says "truncated".
# Bounds the socket write buffer if a client stops reading mid-apply.
MAX_STREAMED_BYTES = 4 * 1024 * 1024

# Mode 0660: root writes, the configured group reads. The group is the operator's
# and is set by the module, not guessed here.
SOCKET_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP


class HostdServer:
    """Serves ``HostdEngine`` over a unix socket."""

    def __init__(
        self,
        engine: HostdEngine,
        *,
        secret: str,
        socket_path: Path,
        group: str | None = None,
    ) -> None:
        if not secret:
            # Fail closed. A helper with no credential would accept anything
            # running as any user that can reach the socket.
            raise ValueError("hostd refuses to start without a shared secret")
        self._engine = engine
        self._secret = secret
        self._path = Path(socket_path)
        self._group = group
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # A socket left behind by a killed helper would make bind fail; it is
        # ours by definition (the module gives us a private RuntimeDirectory).
        if self._path.exists():
            self._path.unlink()
        self._server = await asyncio.start_unix_server(self._handle, path=self._path)
        os.chmod(self._path, SOCKET_MODE)
        if self._group:
            try:
                gid = grp.getgrnam(self._group).gr_gid
            except KeyError:
                raise ValueError(
                    f"hostd was told to expose its socket to group {self._group!r}, "
                    "which does not exist on this host"
                ) from None
            os.chown(self._path, -1, gid)
        logger.info("hostd: listening on %s", self._path)

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def aclose(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._path.unlink(missing_ok=True)

    # --- connection handling ---------------------------------------------

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            await self._dispatch(reader, writer)
        except (ConnectionResetError, BrokenPipeError):
            logger.info("hostd: caller went away")
        except Exception:  # noqa: BLE001 - one bad connection must not stop the helper
            logger.exception("hostd: connection failed")
            await self._send(writer, ErrorFrame(code=ErrorCode.INTERNAL, detail=""))
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError):
                pass

    async def _read_request(self, reader: asyncio.StreamReader) -> Request | None:
        try:
            line = await reader.readuntil(b"\n")
        except asyncio.LimitOverrunError:
            return None
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return None
        if not line or len(line) > MAX_REQUEST_BYTES:
            return None
        try:
            return Request.model_validate_json(line)
        except ValidationError:
            return None

    async def _dispatch(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        request = await self._read_request(reader)
        if request is None:
            await self._send(
                writer,
                ErrorFrame(
                    code=ErrorCode.BAD_REQUEST,
                    detail="not a request this helper understands",
                ),
            )
            return
        if not hmac.compare_digest(request.secret, self._secret):
            # Recorded: an unauthenticated caller on this socket is either a
            # misconfiguration or something on this host trying the door.
            self._engine.record_refusal(
                "a caller presented no valid secret on the hostd socket"
            )
            logger.warning("hostd: refused an unauthenticated caller")
            await self._send(
                writer,
                ErrorFrame(code=ErrorCode.UNAUTHORIZED, detail="invalid credentials"),
            )
            return

        try:
            await self._run_verb(request, reader, writer)
        except HostdRefusal as refusal:
            await self._send(
                writer, ErrorFrame(code=refusal.code, detail=refusal.detail)
            )

    async def _run_verb(
        self,
        request: Request,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if request.verb is Verb.HELLO:
            await self._send(writer, self._engine.hello())
            return
        if request.verb is Verb.AUDIT_TAIL:
            await self._send(writer, self._engine.audit_tail(request.limit))
            return
        if request.verb is Verb.LIST_PENDING:
            await self._send(writer, self._engine.pending())
            return
        if request.verb is Verb.PROPOSE:
            if request.kind is None:
                raise HostdRefusal(
                    ErrorCode.BAD_REQUEST, "propose needs an action kind"
                )
            view = await self._engine.propose(
                request.kind, request.args, request.requester
            )
            await self._send(writer, ProposalFrame(proposal=view))
            return
        if request.verb is Verb.DENY:
            self._engine.deny(
                request.proposal_id,
                operator=request.approved_by,
                reason=request.reason,
            )
            denied = self._engine.proposal(request.proposal_id)
            if denied is None:
                raise HostdRefusal(ErrorCode.NOT_FOUND, "no such proposal")
            await self._send(writer, ProposalFrame(proposal=denied))
            return
        if request.verb is Verb.CANCEL:
            # A cancel with no apply in flight is a no-op, answered honestly
            # rather than pretended into a success.
            raise HostdRefusal(
                ErrorCode.BAD_REQUEST,
                "cancel is sent on the connection of a live apply",
            )
        if request.verb is Verb.APPLY:
            await self._apply(request, reader, writer)
            return
        # Explicit, not a fall-through. The enum makes an unknown verb
        # unreachable today, but "a verb this dispatcher does not recognise
        # defaults to EXECUTING something" is the wrong shape to leave in this
        # file specifically (review round 1, R1.14).
        raise HostdRefusal(
            ErrorCode.BAD_REQUEST, f"this helper has no handler for {request.verb}"
        )

    async def _apply(
        self,
        request: Request,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        streamed = 0
        truncated = False

        def sink(stream: str, text: str) -> None:
            nonlocal streamed, truncated
            if truncated:
                return
            frame = encode(OutputFrame(stream=stream, text=text))
            streamed += len(frame)
            if streamed > MAX_STREAMED_BYTES:
                truncated = True
                writer.write(
                    encode(
                        OutputFrame(
                            stream="stderr",
                            text=(
                                "\n[output truncated: past the byte cap this "
                                "helper streams; the command is still running "
                                "and its result frame is authoritative]\n"
                            ),
                        )
                    )
                )
                return
            writer.write(frame)

        apply_task = asyncio.ensure_future(
            self._engine.apply(
                request.proposal_id,
                on_output=sink,
                approved_by=request.approved_by,
            )
        )
        watcher = asyncio.ensure_future(self._watch_for_cancel(reader))
        done, _pending = await asyncio.wait(
            {apply_task, watcher}, return_when=asyncio.FIRST_COMPLETED
        )
        if watcher in done and apply_task not in done:
            apply_task.cancel()
            try:
                await apply_task
            except asyncio.CancelledError:
                pass
            except HostdRefusal as refusal:
                await self._send(
                    writer, ErrorFrame(code=refusal.code, detail=refusal.detail)
                )
                return
            await self._send(
                writer,
                ErrorFrame(
                    code=ErrorCode.REFUSED,
                    detail=(
                        "cancelled mid-apply; the process group was signalled "
                        "and the cancellation is recorded"
                    ),
                ),
            )
            return
        watcher.cancel()
        result = await apply_task
        await self._send(writer, result)

    async def _watch_for_cancel(self, reader: asyncio.StreamReader) -> None:
        """Resolve when the caller cancels or simply goes away.

        EOF counts. A caller that dropped its connection is not watching the
        output and cannot act on the result, so continuing to run a root command
        on its behalf is the wrong default.
        """
        while True:
            try:
                line = await reader.readuntil(b"\n")
            except (
                asyncio.IncompleteReadError,
                asyncio.LimitOverrunError,
                ConnectionResetError,
            ):
                return
            if not line:
                return
            try:
                follow_up = Request.model_validate_json(line)
            except ValidationError:
                continue
            if not hmac.compare_digest(follow_up.secret, self._secret):
                continue
            if follow_up.verb is Verb.CANCEL:
                return

    async def _send(self, writer: asyncio.StreamWriter, frame: BaseModel) -> None:
        writer.write(encode(frame))
        try:
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            logger.info("hostd: caller closed before reading the answer")
