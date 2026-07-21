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

A2 ships ``CodexBackend`` (exec + app_server) and ``MockBackend``; A2b adds a
``claude`` backend behind the SAME interface, which is what proves the interface
is not accidentally codex-shaped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Iterable,
    Iterator,
    Literal,
    Protocol,
    runtime_checkable,
)

from pydantic import BaseModel

from .agent import (
    AgentReply,
    AgentUnavailable,
    StreamDone,
    StreamError,
    StreamEvent,
    StreamTextDelta,
    StreamTool,
    ToolCall,
    _stream_app_server,
    _stream_codex_exec,
)
from .config import Settings, canonical_backend
from .logsetup import truncate
from .sessions import (
    TokenUsage,
    TranscriptMessage,
    read_context,
    read_transcript,
    resolve_codex_home,
    rollout_mtime,
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


def _codex_sandbox_for(mode: str) -> str:
    return _CODEX_SANDBOX.get(mode, "read-only")


def _claude_permission_mode_for(mode: str) -> str:
    return _CLAUDE_PERMISSION.get(mode, "default")


CodexMode = Literal["exec", "app_server"]


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
    ) -> AsyncIterator[StreamEvent]:
        """Run one turn in ``cwd``, resuming ``session_id`` if given; yield events.

        ``permission_mode`` is the agent's write posture (manual|edit|auto),
        mapped per backend to codex's ``--sandbox`` / claude's
        ``--permission-mode``. Default ``manual`` = read-only.
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


class CodexBackend:
    """The "codex" backend. Uses codex's ``app_server`` runner (token streaming)
    by default; the turn-level ``exec`` runner is retained internally but is no
    longer a user-facing choice. ``name`` is the friendly id "codex" regardless of
    mode; ``stream`` delegates to the agent.py runners (with the A0 ``cwd`` seam);
    ``read_status`` reads the rollout via sessions.py."""

    def __init__(self, mode: CodexMode = "app_server") -> None:
        self._mode: CodexMode = mode
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
    ) -> AsyncIterator[StreamEvent]:
        runner = (
            _stream_app_server if self._mode == "app_server" else _stream_codex_exec
        )
        sandbox = _codex_sandbox_for(permission_mode)
        async for event in runner(
            settings, prompt, session_id, image_paths, cwd=cwd, sandbox=sandbox
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
        return read_transcript(resolve_codex_home(settings), session_id)


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


def _claude_stream_args(
    claude_bin: str,
    prompt: str,
    permission_mode: str,
    session_id: str | None,
    claude_home: Path,
) -> list[str]:
    """Build the ``claude -p`` argument list for one turn. ``--resume`` is added
    ONLY when the session actually exists on disk - resuming an unknown session
    (a stale/deleted id, or one from a different backend after a backend switch)
    makes claude fail the whole turn with ``error_during_execution`` ("No
    conversation found with session ID"). An unresumable id just starts fresh.
    Pure (a filesystem lookup, no subprocess), so it is unit-testable."""
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
    if session_id and _find_claude_session(claude_home, session_id) is not None:
        args += ["--resume", session_id]
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
    ) -> AsyncIterator[StreamEvent]:
        # image attachments are an A3 follow-up. The permission mode maps to
        # claude's --permission-mode (default/acceptEdits/bypassPermissions); the
        # exact read-only enforcement of "default" in headless mode is weaker than
        # codex's sandbox and should be verified live when write modes are used.
        claude_bin = self._resolve_bin(settings)
        args = _claude_stream_args(
            claude_bin,
            prompt,
            permission_mode,
            session_id,
            resolve_claude_home(settings),
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
        )
        try:
            assert proc.stdout is not None
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                for event in parse_claude_stream([raw.decode(errors="replace")]):
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
    if canonical == "mock":
        return MockBackend()
    raise ValueError(f"unknown backend: {name!r}")
