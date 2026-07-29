"""The ``AgentBackend`` interface: one seam over codex and (later, A2b) claude.

The orchestrator, the run supervisor, the dashboard and the store all speak to an
agent through this interface, so nothing above it branches on which backend an
agent uses (spike tasks/20260720-221748 decisions 1 and 4). A backend does two
things:

- ``stream(...)`` runs one turn scoped to the agent's project ``cwd``, resuming a
  ``session_id`` when given, and yields normalized ``StreamEvent``s (the same
  events the A0 event bus fans out).
- ``read_status(...)`` returns a READ-ONLY snapshot of a session's progress
  derived from its durable log (for codex, the rollout JSONL) - the "what is this
  agent doing" half. The live run-state (queued/running/done) comes from the A0
  Supervisor and is merged with this in A3/A5.

A2 shipped ``CodexBackend`` (originally exec + app_server; the exec mode was
dropped 20260721-152746, so it is now app_server-only) and ``MockBackend``; A2b
adds a ``claude`` backend behind the SAME interface, which is what proves the
interface is not accidentally codex-shaped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Iterable,
    Iterator,
    Protocol,
    runtime_checkable,
)

import httpx
from pydantic import BaseModel

from .agent import (
    STREAM_READ_LIMIT,
    AgentReply,
    AgentUnavailable,
    StreamDone,
    StreamError,
    StreamEvent,
    StreamTextDelta,
    StreamTool,
    ToolCall,
    _stream_app_server,
    agent_subprocess_env,
    scufris_mcp_servers,
)
from .config import Settings, canonical_backend
from .logsetup import truncate
from .opencode_client import (
    Message,
    ModelRef,
    OpencodeClient,
    OpencodeError,
    OpencodeStaleSessionError,
    SendMessageRequest,
    TextPartInput,
)
from .reasoning_store import ReasoningStore
from .sessions import (
    SessionContext,
    SessionInfo,
    TokenUsage,
    TranscriptMessage,
    merge_reasoning,
    read_context,
    read_transcript,
    resolve_codex_home,
    rollout_mtime,
)
from .sessions import (
    delete_session as codex_delete_session,
)

logger = logging.getLogger(__name__)

# How much of the last assistant message to keep in a status snapshot.
_LAST_MESSAGE_PREVIEW = 280

# Permission mode -> per-backend flag (values verified live via `--help`).
_CODEX_SANDBOX = {
    "manual": "read-only",
    "edit": "workspace-write",
    "auto": "danger-full-access",
}
_CLAUDE_PERMISSION = {
    "manual": "default",
    "edit": "acceptEdits",
    "auto": "bypassPermissions",
}
# Permission mode -> opencode per-request `tools` enable/disable map. opencode's
# approval flow ("ask") has no answerer on a headless server, so the safe lever is
# tool AVAILABILITY: a disabled tool cannot be called. manual disables all mutating
# tools (read-only); edit allows edits but not shell; auto leaves everything on
# (empty map). See tasks/20260722-135520/NOTES.md.
_OPENCODE_MUTATING_TOOLS = ("edit", "write", "patch", "bash")
_OPENCODE_PERMISSION: dict[str, dict[str, bool]] = {
    "manual": {tool: False for tool in _OPENCODE_MUTATING_TOOLS},
    "edit": {"bash": False},
    "auto": {},
}


def _opencode_tools_for(mode: str) -> dict[str, bool]:
    return _OPENCODE_PERMISSION.get(mode, _OPENCODE_PERMISSION["manual"])


def _codex_sandbox_for(mode: str) -> str:
    return _CODEX_SANDBOX.get(mode, "read-only")


def _claude_permission_mode_for(mode: str) -> str:
    return _CLAUDE_PERMISSION.get(mode, "default")


class BackendStatus(BaseModel):
    """A read-only snapshot of one agent session's progress, normalized across
    backends. Derived from the durable log, not the live run - ``state`` is left
    to the Supervisor and merged in later."""

    session_id: str
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    context_window: int = 0
    last_message: str | None = None
    updated_at: float | None = None


def _context_from_status(status: BackendStatus | None) -> SessionContext | None:
    """Map a ``BackendStatus`` onto a ``SessionContext`` for backends that expose no
    richer per-session context breakdown than their status snapshot (claude,
    opencode). The codex-only cached/reasoning/total token axes stay 0; window is
    whatever the status reports (0 when the backend does not surface one)."""
    if status is None:
        return None
    return SessionContext(
        session_id=status.session_id,
        context_window=status.context_window,
        input_tokens=status.input_tokens,
        output_tokens=status.output_tokens,
        turn_count=status.turns,
        tool_call_count=status.tool_calls,
    )


@runtime_checkable
class AgentBackend(Protocol):
    """What the orchestrator/supervisor depend on; implementations are swappable."""

    name: str

    def stream(
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
        """Run one turn in ``cwd``, resuming ``session_id`` if given; yield events.

        ``permission_mode`` is the agent's write posture (manual|edit|auto),
        mapped per backend to codex's ``--sandbox`` / claude's
        ``--permission-mode``. Default ``manual`` = read-only.

        ``is_orchestrator`` marks the turn as the landing orchestrator's, which
        selects the orchestrator role of the scufris MCP server and its steering
        (codex backend). ``agent_id`` is the caller's own id, passed to a regular
        (non-orchestrator) agent's scufris server so its ``request_input`` callback
        can address itself. Both are a no-op for backends without MCP wiring
        (claude, mock).
        """
        ...

    def read_status(
        self, settings: Settings, session_id: str | None
    ) -> BackendStatus | None:
        """A read-only progress snapshot for ``session_id``, or None if unreadable."""
        ...

    def read_transcript(
        self, settings: Settings, session_id: str | None
    ) -> list[TranscriptMessage]:
        """The session's past messages (for rebuilding the chat), oldest-first;
        empty when ``session_id`` is unset or the session cannot be read."""
        ...

    def read_context(
        self, settings: Settings, session_id: str | None
    ) -> SessionContext | None:
        """The session's context snapshot (window + token usage + counts), or None
        when unset/unreadable. codex reads the rich rollout breakdown; other
        backends map what their status snapshot exposes (window 0 when the backend
        does not report one)."""
        ...

    async def delete_session(self, settings: Settings, session_id: str | None) -> bool:
        """Delete the session's provider-side record, returning True if removed.
        Codex unlinks the rollout, claude the transcript file, opencode calls the
        daemon; a backend with no provider delete returns False. Never raises - a
        False just means the registry forget (caller side) is the only cleanup."""
        ...


class CodexBackend:
    """The "codex" backend: codex's ``app_server`` runner (token streaming). The
    turn-level ``exec`` runner is no longer a per-agent choice (dropped
    20260721-152746); ``stream`` always uses app_server. ``name`` is the friendly
    id "codex"; ``read_status`` reads the rollout via sessions.py."""

    def __init__(self) -> None:
        self.name: str = "codex"

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
        sandbox = _codex_sandbox_for(permission_mode)
        async for event in _stream_app_server(
            settings,
            prompt,
            session_id,
            image_paths,
            cwd=cwd,
            sandbox=sandbox,
            is_orchestrator=is_orchestrator,
            agent_id=agent_id,
        ):
            yield event

    def read_status(
        self, settings: Settings, session_id: str | None
    ) -> BackendStatus | None:
        if not session_id:
            return None
        home = resolve_codex_home(settings)
        ctx = read_context(home, session_id)
        if ctx is None:
            return None
        last_message: str | None = None
        for msg in reversed(read_transcript(home, session_id)):
            if msg.role == "assistant" and msg.text.strip():
                last_message = truncate(msg.text.strip(), _LAST_MESSAGE_PREVIEW)
                break
        return BackendStatus(
            session_id=session_id,
            turns=ctx.turn_count,
            tool_calls=ctx.tool_call_count,
            input_tokens=ctx.input_tokens,
            output_tokens=ctx.output_tokens,
            context_window=ctx.context_window,
            last_message=last_message,
            updated_at=rollout_mtime(home, session_id),
        )

    def read_transcript(
        self, settings: Settings, session_id: str | None
    ) -> list[TranscriptMessage]:
        if not session_id:
            return []
        messages = read_transcript(resolve_codex_home(settings), session_id)
        # Reasoning is not on disk (encrypted blob), so re-hydrate the "thinking"
        # spoilers from scufris's own sidecar - merged here (not in the pure
        # rollout reader) because the sidecar lives under state_dir, not codex_home.
        merge_reasoning(messages, ReasoningStore(settings).read(session_id))
        return messages

    def read_context(
        self, settings: Settings, session_id: str | None
    ) -> SessionContext | None:
        # The rich rollout reader (keeps cached/reasoning/total + window).
        return read_context(resolve_codex_home(settings), session_id)

    async def delete_session(self, settings: Settings, session_id: str | None) -> bool:
        return codex_delete_session(resolve_codex_home(settings), session_id)


class MockBackend:
    """An in-process backend for tests/offline demos - no codex, no network."""

    name: str = "mock"

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
        yield StreamTextDelta(delta=f"[mock] {prompt}")
        yield StreamDone(
            reply=AgentReply(text=f"[mock reply] {prompt}"),
            session_id=session_id or "mock-session",
        )

    def read_status(
        self, settings: Settings, session_id: str | None
    ) -> BackendStatus | None:
        if not session_id:
            return None
        return BackendStatus(
            session_id=session_id, turns=1, last_message="[mock] running"
        )

    def read_transcript(
        self, settings: Settings, session_id: str | None
    ) -> list[TranscriptMessage]:
        # The in-process mock keeps no on-disk transcript.
        return []

    def read_context(
        self, settings: Settings, session_id: str | None
    ) -> SessionContext | None:
        # The in-process mock keeps no context snapshot.
        return None

    async def delete_session(self, settings: Settings, session_id: str | None) -> bool:
        # Nothing on disk / no daemon to delete from.
        return False


# --- claude (Claude Code headless) backend ----------------------------------
#
# Probed shape of `claude -p <prompt> --output-format stream-json --verbose`
# (tasks/20260720-223938/NOTES.md): a JSONL stream of
#   {"type":"system","subtype":"init","session_id",...}
#   {"type":"assistant","message":{"content":[{"type":"text"|"tool_use",...}],
#                                  "usage":{...}}}
#   {"type":"result","subtype":"success"|...,"result":<text>,"session_id",...}
# and a session transcript at <claude_home>/projects/<cwd-hash>/<session_id>.jsonl
# (found by session-id glob, so read_status needs no cwd - same as codex).


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
    ``approval_mode="approve"`` (see DECISION.md in tasks/20260723-201851).
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
    filename, id known before the turn; part 2). Resume WINS: the two are never
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
        # image attachments are an A3 follow-up. The permission mode maps to
        # claude's --permission-mode (default/acceptEdits/bypassPermissions); the
        # exact read-only enforcement of "default" in headless mode is weaker than
        # codex's sandbox and should be verified live when write modes are used.
        claude_bin = self._resolve_bin(settings)
        claude_home = resolve_claude_home(settings)
        # On a FRESH turn (nothing resumable), mint the session id ourselves and
        # pass it as --session-id, so the id is deterministic and known before the
        # turn instead of scraped from the result frame - and still recoverable if
        # the result frame omits it (see the StreamDone substitution below). A
        # resume keeps claude's existing id; never feed a stale/foreign id to
        # --session-id (must be a valid UUID). Part 2, DECISION 20260724-111955.
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
            # it (review round 2, R2.1).
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


# --- opencode (opencode serve -> llama.cpp) backend --------------------------
#
# opencode's headless surface is an HTTP daemon (`opencode serve`), so this
# backend drives it over an async HTTP client rather than a stdio subprocess -
# structurally the codex `app_server` shape, not the claude stdin one. v0 runs a
# turn SYNCHRONOUSLY (`send_message` blocks and returns the whole `{info, parts}`
# reply); live token streaming over the daemon's `/event` SSE bus is a deferred
# follow-up. read_status/read_transcript read the session back over
# `GET /session/{id}/message`. See tasks/20260722-135404/SPIKE.md and
# tasks/20260722-135520/NOTES.md.


def _opencode_ms_to_dt(ms: int | None) -> "datetime | None":
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, timezone.utc)


def _opencode_usage(msg: Message) -> TokenUsage | None:
    tokens = msg.info.tokens
    if tokens is None:
        return None
    cache = (tokens.model_extra or {}).get("cache") or {}
    cached = int(cache.get("read", 0)) if isinstance(cache, dict) else 0
    return TokenUsage(
        input_tokens=tokens.input,
        cached_input_tokens=cached,
        output_tokens=tokens.output,
        reasoning_output_tokens=tokens.reasoning,
    )


def _opencode_tool_calls(msg: Message) -> list[ToolCall]:
    return [
        ToolCall(server="opencode", tool=p.tool_name(), status="completed")
        for p in msg.tool_parts()
    ]


def parse_opencode_transcript(messages: list[Message]) -> list[TranscriptMessage]:
    """Fold opencode `{info, parts}` messages into TranscriptMessages, oldest-first.

    A user or assistant message with any text becomes a message; a tool-only
    assistant turn is kept (empty text, carrying its tool calls), mirroring the
    claude transcript parser.
    """
    out: list[TranscriptMessage] = []
    for msg in messages:
        role = msg.info.role
        if role not in ("user", "assistant"):
            continue
        text = msg.text()
        tools = _opencode_tool_calls(msg) if role == "assistant" else []
        if not text and not tools and role == "assistant":
            continue
        out.append(
            TranscriptMessage(
                role=role,
                text=text,
                ts=_opencode_ms_to_dt(msg.info.time.created if msg.info.time else None),
                tool_calls=tools,
                usage=_opencode_usage(msg) if role == "assistant" else None,
            )
        )
    return out


class OpenCodeBackend:
    """The "opencode" backend: drives a running `opencode serve` daemon over HTTP.

    ``name`` is "opencode". The daemon URL/password/model/provider come from
    ``Settings`` (``opencode_url`` etc). ``_make_client`` is a seam the tests
    monkeypatch to inject a fake client.
    """

    name: str = "opencode"

    def _make_client(self, settings: Settings) -> OpencodeClient:
        return OpencodeClient(
            settings.opencode_url,
            settings.opencode_password,
            timeout=settings.agent_timeout_seconds,
        )

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
        # cwd is not used: the daemon's working dir is fixed at `opencode serve`
        # launch, not per turn (unlike codex/claude which take cwd per subprocess).
        # Image attachments are a follow-up (FilePartInput), like the claude path.
        request = SendMessageRequest(
            parts=[TextPartInput(text=prompt)],
            model=ModelRef(
                providerID=settings.opencode_provider, modelID=settings.opencode_model
            ),
            tools=_opencode_tools_for(permission_mode) or None,
        )
        client = self._make_client(settings)
        try:
            reply = await self._send(client, session_id, request, agent_id=agent_id)
        except OpencodeError as exc:
            yield StreamError(detail=f"opencode: {exc}")
            return
        finally:
            await client.close()

        msg, new_session_id = reply
        for call in _opencode_tool_calls(msg):
            yield StreamTool(tool=call)
        text = msg.text()
        if text:
            yield StreamTextDelta(delta=text)
        yield StreamDone(
            reply=AgentReply(
                text=text.strip(),
                status="completed",
                tool_calls=_opencode_tool_calls(msg),
                usage=_opencode_usage(msg),
            ),
            session_id=new_session_id,
        )

    async def _send(
        self,
        client: OpencodeClient,
        session_id: str | None,
        request: SendMessageRequest,
        *,
        agent_id: str = "",
    ) -> tuple[Message, str]:
        """Resolve/create a session and run one turn; recreate once on a stale id.

        A newly created session is tagged with ``metadata={"agent_id": ...}`` so
        ownership is recorded on the provider side (part 2). A resumed session is
        left untouched (it was tagged when first created)."""
        metadata = {"agent_id": agent_id} if agent_id else None
        sid = session_id or (await client.create_session(metadata=metadata)).id
        try:
            return await client.send_message(sid, request), sid
        except OpencodeStaleSessionError:
            # The id we were handed is gone (deleted, or a cross-backend id after a
            # backend switch); start a fresh session and retry once.
            fresh = (await client.create_session(metadata=metadata)).id
            return await client.send_message(fresh, request), fresh

    def read_status(
        self, settings: Settings, session_id: str | None
    ) -> BackendStatus | None:
        if not session_id:
            return None
        messages = self._read_messages(settings, session_id)
        if messages is None:
            return None
        turns = sum(1 for m in messages if m.info.role == "user")
        tool_calls = sum(len(m.tool_parts()) for m in messages)
        last_message: str | None = None
        input_tokens = output_tokens = 0
        updated_at: datetime | None = None
        for msg in messages:
            if msg.info.time and msg.info.time.created:
                updated_at = _opencode_ms_to_dt(msg.info.time.created)
            if msg.info.role == "assistant":
                if msg.text().strip():
                    last_message = truncate(msg.text().strip(), _LAST_MESSAGE_PREVIEW)
                usage = _opencode_usage(msg)
                if usage is not None:
                    input_tokens = usage.input_tokens or input_tokens
                    output_tokens = usage.output_tokens or output_tokens
        return BackendStatus(
            session_id=session_id,
            turns=turns,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context_window=0,
            last_message=last_message,
            updated_at=updated_at.timestamp() if updated_at else None,
        )

    def read_transcript(
        self, settings: Settings, session_id: str | None
    ) -> list[TranscriptMessage]:
        if not session_id:
            return []
        messages = self._read_messages(settings, session_id)
        return parse_opencode_transcript(messages) if messages else []

    def read_context(
        self, settings: Settings, session_id: str | None
    ) -> SessionContext | None:
        # opencode exposes no per-session context window; map read_status.
        return _context_from_status(self.read_status(settings, session_id))

    async def delete_session(self, settings: Settings, session_id: str | None) -> bool:
        """Delete the session on the daemon via ``OpencodeClient``. Any failure ->
        False (never raises), so the registry forget is the fallback."""
        if not session_id:
            return False
        client = self._make_client(settings)
        try:
            return await client.delete_session(session_id)
        except OpencodeError:
            return False
        finally:
            await client.close()

    def _read_messages(
        self, settings: Settings, session_id: str
    ) -> list[Message] | None:
        """Fetch a session's messages via the daemon; None if it cannot be read.

        read_status/read_transcript are SYNCHRONOUS (the AgentBackend protocol),
        and their FastAPI/FastMCP handlers are sync `def` (run in a threadpool, not
        on the event loop), so a plain blocking httpx read is the simplest correct
        choice - it mirrors codex/claude's blocking file reads at the same call
        sites (app.py agent_run_status/agent_transcript, mcp_server agent_status).
        Any failure -> None (never crash a snapshot).
        """
        auth = (
            httpx.BasicAuth("", settings.opencode_password)
            if settings.opencode_password
            else None
        )
        url = f"{settings.opencode_url.rstrip('/')}/session/{session_id}/message"
        try:
            resp = httpx.get(url, auth=auth, timeout=10.0)
            if resp.status_code != 200:
                return None
            return [Message.model_validate(item) for item in resp.json()]
        except (httpx.HTTPError, ValueError):
            return None


# Cap a switcher title to the same length codex's own lister used (sessions.py
# `_TITLE_MAX`), so titles do not vary by which path built them.
_SESSION_TITLE_MAX = 80


def session_info(
    backend: AgentBackend, settings: Settings, session_id: str
) -> SessionInfo | None:
    """A ``SessionInfo`` for the orchestrator switcher, hydrated backend-agnostically
    from the session's OWN store (part 1): title = its first user message,
    ``started_at`` = that message's timestamp, ``updated_at`` = the backend status
    snapshot's mtime. Returns None when the id resolves to no readable session, so
    a stale/foreign id drops out of the list instead of showing as "(untitled)".

    ``git_branch``/``cwd`` are left unset: they are codex rollout-meta niceties
    the generic (transcript + status) path does not parse, and the switcher UI
    does not depend on them. Reads the transcript and a status snapshot, so it is
    heavier than a head-only scan - fine for a short switcher list (see
    tasks/20260724-111947/DECISION.md Consequences)."""
    messages = backend.read_transcript(settings, session_id)
    status = backend.read_status(settings, session_id)
    if not messages and status is None:
        return None
    title = ""
    started_at: datetime | None = None
    for message in messages:
        if message.role == "user" and message.text.strip():
            title = message.text.strip()
            started_at = message.ts
            break
    updated_at: datetime | None = None
    if status is not None and status.updated_at is not None:
        updated_at = datetime.fromtimestamp(status.updated_at, timezone.utc)
    return SessionInfo(
        id=session_id,
        title=(title or "(untitled)")[:_SESSION_TITLE_MAX],
        started_at=started_at,
        updated_at=updated_at,
    )


def get_backend(name: str) -> AgentBackend:
    """Resolve a backend by (possibly legacy) name; unknown raises.

    Legacy codex modes (`app_server`/`exec`) and `codex` all resolve to the codex
    backend (app_server runner). `mock` always RESOLVES (an already-persisted mock
    agent must still run); whether a mock agent may be CREATED is gated separately
    by `enable_mock_backend` in the store.
    """
    canonical = canonical_backend(name)
    if canonical == "codex":
        return CodexBackend()
    if canonical == "claude":
        return ClaudeBackend()
    if canonical == "opencode":
        return OpenCodeBackend()
    if canonical == "mock":
        return MockBackend()
    raise ValueError(f"unknown backend: {name!r}")
