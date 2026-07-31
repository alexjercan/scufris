"""The claude adapter: Claude Code headless, plus its stream and transcript parsers.

Probed shape of `claude -p <prompt> --output-format stream-json --verbose` -
a JSONL stream of

    {"type":"system","subtype":"init","session_id",...}
    {"type":"assistant","message":{"content":[{"type":"text"|"tool_use",...}],
                                   "usage":{...}}}
    {"type":"result","subtype":"success"|...,"result":<text>,"session_id",...}

and a session transcript at ``<claude_home>/projects/<cwd-hash>/<session_id>.jsonl``
(found by session-id glob, so read_status needs no cwd - same as codex).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Iterator

from ..agent import (
    STREAM_READ_LIMIT,
    AgentReply,
    AgentUnavailable,
    StreamDone,
    StreamError,
    StreamEvent,
    StreamTextDelta,
    StreamTool,
    ToolCall,
    agent_subprocess_env,
    scufris_mcp_servers,
)
from ..config import Settings
from ..logsetup import truncate
from ..sessions import SessionContext, TokenUsage, TranscriptMessage
from .base import _LAST_MESSAGE_PREVIEW, BackendStatus, _context_from_status

logger = logging.getLogger(__name__)

#: Permission mode -> claude `--permission-mode` value (verified live via `--help`).
_CLAUDE_PERMISSION = {
    "manual": "default",
    "edit": "acceptEdits",
    "auto": "bypassPermissions",
}


def _claude_permission_mode_for(mode: str) -> str:
    return _CLAUDE_PERMISSION.get(mode, "default")


def resolve_claude_home(settings: Settings) -> Path:
    """Where Claude Code stores sessions (settings override or ``~/.claude``)."""
    return settings.claude_home or Path.home() / ".claude"


def parse_claude_stream(lines: Iterable[str]) -> Iterator[StreamEvent]:
    """Map claude ``stream-json`` lines to normalized ``StreamEvent``s.

    Pure (no I/O), so it is tested directly against captured real output. Unknown
    line types are ignored so a new event kind is additive, not a failure.
    """
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        kind = obj.get("type")
        if kind == "assistant":
            message = obj.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            for block in content or []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text" and block.get("text"):
                    yield StreamTextDelta(delta=str(block["text"]))
                elif btype == "tool_use":
                    yield StreamTool(
                        tool=ToolCall(
                            server="claude",
                            tool=str(block.get("name") or "tool"),
                            status="completed",
                        )
                    )
        elif kind == "result":
            if obj.get("is_error") or obj.get("subtype") != "success":
                # Some failure subtypes (e.g. error_max_turns) carry no `result`
                # text; include the subtype so the error is diagnosable.
                detail = str(
                    obj.get("result") or obj.get("subtype") or "claude turn failed"
                )
                yield StreamError(detail=detail)
            else:
                yield StreamDone(
                    reply=AgentReply(text=str(obj.get("result") or "")),
                    session_id=obj.get("session_id"),
                )


def _find_claude_session(claude_home: Path, session_id: str) -> Path | None:
    """Locate a claude session transcript by id, under any project dir."""
    projects = claude_home / "projects"
    if not projects.is_dir():
        return None
    matches = list(projects.rglob(f"{session_id}.jsonl"))
    return matches[0] if matches else None


def _scufris_claude_args(
    settings: Settings,
    *,
    is_orchestrator: bool,
    agent_id: str,
    orch_session_id: str = "",
) -> list[str]:
    """The claude flags registering this turn's scufris MCP servers, or ``[]`` when
    the turn gets none (tools off / no audience).

    Formats the shared ``scufris_mcp_servers`` core (the same command/args/env
    codex gets - an orchestrator turn's ``scufris`` + ``den``, a sub-agent turn's
    ``agent``) into an INLINE ``--mcp-config`` JSON blob, plus ``--strict-mcp-config``
    (scope the turn to exactly our servers, ignoring project ``.mcp.json`` / global
    config) and an ``--allowedTools mcp__<id>__*`` wildcard PER registered server so
    the unattended turn auto-approves those tools instead of hanging on an approval
    prompt. The whole-server wildcard is SAFE because the audience split is physical
    (only the turn's own servers are registered) - mirroring codex's whole-server
    ``approval_mode="approve"``.
    ``--mcp-config`` is VARIADIC/greedy (it eats following tokens as more config
    paths until the next flag), so it MUST be followed by a flag; here
    ``--strict-mcp-config`` bounds it.
    """
    servers = scufris_mcp_servers(
        settings,
        is_orchestrator=is_orchestrator,
        agent_id=agent_id,
        orch_session_id=orch_session_id,
    )
    if not servers:
        return []
    config = {
        "mcpServers": {
            server.server_id: {
                "command": server.command,
                "args": list(server.args),
                "env": server.env,
            }
            for server in servers
        }
    }
    args = ["--mcp-config", json.dumps(config), "--strict-mcp-config"]
    for server in servers:
        args += ["--allowedTools", f"mcp__{server.server_id}__*"]
    return args


def _claude_stream_args(
    claude_bin: str,
    prompt: str,
    permission_mode: str,
    session_id: str | None,
    claude_home: Path,
    settings: Settings,
    *,
    is_orchestrator: bool = False,
    agent_id: str = "",
    new_session_id: str | None = None,
    resumable: bool | None = None,
) -> list[str]:
    """Build the ``claude -p`` argument list for one turn.

    ``--resume`` is added ONLY when the session actually exists on disk - resuming
    an unknown session (a stale/deleted id, or one from a different backend after
    a backend switch) makes claude fail the whole turn with
    ``error_during_execution`` ("No conversation found with session ID"). When the
    turn is NOT a resume and ``new_session_id`` is given, it is passed as
    ``--session-id`` so scufris - not claude - picks the id (deterministic
    filename, id known before the turn). Resume WINS: the two are never
    passed together, and a stale/foreign id is never fed to ``--session-id`` (it
    must be a valid UUID, and a codex id would be wrong) - the caller mints a
    fresh UUID for that.

    The scufris MCP flags (``_scufris_claude_args``) ride EVERY turn's argv, so a
    resumed turn re-loads the turn's servers the same as a fresh one - the args are
    rebuilt per turn, the way codex re-sends its sandbox per turn. Pure (a
    filesystem lookup + a settings read, no subprocess), so it is unit-testable."""
    args = [
        claude_bin,
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        _claude_permission_mode_for(permission_mode),
    ]
    # ``resumable`` lets the caller pass a decision it already computed (``stream``
    # scans once to choose whether to mint), avoiding a second disk scan here; when
    # None (the pure-function unit tests) it is derived locally.
    if resumable is None:
        resumable = (
            bool(session_id)
            and _find_claude_session(claude_home, session_id or "") is not None
        )
    # The orchestrator's current chat = the session this turn resumes (empty on a
    # fresh turn); rides the scufris env so a spawned child gets stamped with it.
    orch_session_id = (
        session_id if (is_orchestrator and resumable and session_id) else ""
    )
    args += _scufris_claude_args(
        settings,
        is_orchestrator=is_orchestrator,
        agent_id=agent_id,
        orch_session_id=orch_session_id,
    )
    if resumable and session_id:
        args += ["--resume", session_id]
    elif new_session_id:
        args += ["--session-id", new_session_id]
    return args


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        text = path.read_text()
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


def parse_claude_transcript(
    objs: Iterable[dict[str, Any]], limit: int = 200
) -> list[TranscriptMessage]:
    """Map claude session JSONL objects to TranscriptMessages (oldest-first).

    Pure (no I/O), so it is tested against captured-shape objects. A real user
    turn is a string-content ``user`` entry (tool_result turns carry a list and
    are skipped); an assistant turn concatenates its text blocks and carries the
    tools it ran + the turn's token usage. Empty assistant frames (a bare
    tool_result ack) are dropped.
    """
    messages: list[TranscriptMessage] = []
    for obj in objs:
        kind = obj.get("type")
        message = obj.get("message")
        if kind == "user" and isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                messages.append(TranscriptMessage(role="user", text=content))
        elif kind == "assistant" and isinstance(message, dict):
            texts: list[str] = []
            tools: list[ToolCall] = []
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text"):
                    texts.append(str(block["text"]))
                elif block.get("type") == "tool_use":
                    tools.append(
                        ToolCall(
                            server="claude",
                            tool=str(block.get("name") or "tool"),
                            status="completed",
                        )
                    )
            usage_obj = message.get("usage")
            usage: TokenUsage | None = None
            if isinstance(usage_obj, dict):
                usage = TokenUsage(
                    input_tokens=int(usage_obj.get("input_tokens") or 0),
                    output_tokens=int(usage_obj.get("output_tokens") or 0),
                )
            if texts or tools:
                messages.append(
                    TranscriptMessage(
                        role="assistant",
                        text="\n".join(texts),
                        tool_calls=tools,
                        usage=usage,
                    )
                )
    return messages[-limit:]


class ClaudeBackend:
    """Claude Code headless behind the interface. ``stream`` shells out to
    ``claude -p ... --output-format stream-json``; ``read_status`` reads the
    session transcript found by id under ``<claude_home>/projects``."""

    name: str = "claude"

    def _resolve_bin(self, settings: Settings) -> str:
        claude_bin = settings.claude_bin or shutil.which("claude")
        if not claude_bin:
            raise AgentUnavailable(
                "claude CLI not found. Install Claude Code or set SCUFRIS_CLAUDE_BIN."
            )
        return claude_bin

    async def stream(
        self,
        settings: Settings,
        prompt: str,
        *,
        session_id: str | None = None,
        cwd: str | None = None,
        image_paths: list[str] | None = None,
        permission_mode: str = "manual",
        is_orchestrator: bool = False,
        agent_id: str = "",
    ) -> AsyncIterator[StreamEvent]:
        # TODO: support image attachments on this backend.
        #
        # The permission mode maps to claude's --permission-mode
        # (default/acceptEdits/bypassPermissions); the
        # exact read-only enforcement of "default" in headless mode is weaker than
        # codex's sandbox and should be verified live when write modes are used.
        claude_bin = self._resolve_bin(settings)
        claude_home = resolve_claude_home(settings)
        # On a FRESH turn (nothing resumable), mint the session id ourselves and
        # pass it as --session-id, so the id is deterministic and known before the
        # turn instead of scraped from the result frame - and still recoverable if
        # the result frame omits it (see the StreamDone substitution below). A
        # resume keeps claude's existing id; never feed a stale/foreign id to
        # --session-id (must be a valid UUID).
        resumable = (
            bool(session_id)
            and _find_claude_session(claude_home, session_id or "") is not None
        )
        new_session_id = None if resumable else str(uuid.uuid4())
        args = _claude_stream_args(
            claude_bin,
            prompt,
            permission_mode,
            session_id,
            claude_home,
            settings,
            is_orchestrator=is_orchestrator,
            agent_id=agent_id,
            new_session_id=new_session_id,
            resumable=resumable,
        )
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            # stderr is not consumed; DEVNULL (not PIPE) so a chatty stderr can
            # never fill an undrained pipe buffer and deadlock the turn while we
            # read stdout line-by-line.
            stderr=asyncio.subprocess.DEVNULL,
            cwd=cwd,
            # Never a bare inherit: the child is the model's shell, and this
            # environment is the ONE place a scufris credential is stripped from
            # it.
            env=agent_subprocess_env(settings),
            # A single stream-json frame (a large tool result / file dump) can far
            # exceed asyncio's default 64 KiB readline limit; raise it so such
            # lines stream through instead of raising `ValueError`. Shared with the
            # codex app-server launch. See STREAM_READ_LIMIT.
            limit=STREAM_READ_LIMIT,
        )
        try:
            assert proc.stdout is not None
            while True:
                try:
                    raw = await proc.stdout.readline()
                except ValueError:
                    # An over-STREAM_READ_LIMIT line: surface a clean, diagnosable
                    # StreamError instead of a bare uncaught ValueError.
                    proc.kill()
                    yield StreamError(
                        detail=f"claude line exceeded {STREAM_READ_LIMIT}-byte read limit"
                    )
                    return
                if not raw:
                    break
                for event in parse_claude_stream([raw.decode(errors="replace")]):
                    # Guarantee a fresh turn's StreamDone carries the id we minted,
                    # even if the result frame omitted session_id - so the run is
                    # never recorded without its session id.
                    if (
                        new_session_id is not None
                        and isinstance(event, StreamDone)
                        and not event.session_id
                    ):
                        event = event.model_copy(update={"session_id": new_session_id})
                    yield event
            await proc.wait()
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    def read_status(
        self, settings: Settings, session_id: str | None
    ) -> BackendStatus | None:
        if not session_id:
            return None
        path = _find_claude_session(resolve_claude_home(settings), session_id)
        if path is None:
            return None
        turns = 0
        tools = 0
        last_message: str | None = None
        input_tokens = 0
        output_tokens = 0
        for obj in _iter_jsonl(path):
            kind = obj.get("type")
            message = obj.get("message")
            if kind == "user" and isinstance(message, dict):
                # A real user turn carries a string prompt; tool_result turns
                # carry a list and are not counted as turns.
                if isinstance(message.get("content"), str):
                    turns += 1
            elif kind == "assistant" and isinstance(message, dict):
                for block in message.get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and block.get("text"):
                        last_message = str(block["text"])
                    elif block.get("type") == "tool_use":
                        tools += 1
                usage = message.get("usage")
                if isinstance(usage, dict):
                    input_tokens = int(usage.get("input_tokens") or input_tokens)
                    output_tokens = int(usage.get("output_tokens") or output_tokens)
        try:
            updated_at: float | None = path.stat().st_mtime
        except OSError:
            updated_at = None
        return BackendStatus(
            session_id=session_id,
            turns=turns,
            tool_calls=tools,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context_window=0,
            last_message=(
                truncate(last_message, _LAST_MESSAGE_PREVIEW) if last_message else None
            ),
            updated_at=updated_at,
        )

    def read_transcript(
        self, settings: Settings, session_id: str | None
    ) -> list[TranscriptMessage]:
        if not session_id:
            return []
        path = _find_claude_session(resolve_claude_home(settings), session_id)
        if path is None:
            return []
        return parse_claude_transcript(_iter_jsonl(path))

    def read_context(
        self, settings: Settings, session_id: str | None
    ) -> SessionContext | None:
        # claude exposes no per-session context window; map what read_status has.
        return _context_from_status(self.read_status(settings, session_id))

    async def delete_session(self, settings: Settings, session_id: str | None) -> bool:
        """Unlink the claude transcript file (``<id>.jsonl``). False when the id is
        unset/unknown or the unlink fails - never raises."""
        if not session_id:
            return False
        path = _find_claude_session(resolve_claude_home(settings), session_id)
        if path is None:
            return False
        try:
            path.unlink()
        except OSError:
            logger.warning("claude delete_session %s -> unlink failed", session_id)
            return False
        return True
