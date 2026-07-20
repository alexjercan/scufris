"""The Scufris agent backend.

The agent is what runs tools and chats about the host. It is fronted by the
small ``Agent`` protocol so the harness is swappable; the default implementation
drives the ``codex`` CLI (nixpkgs `codex`, "Sign in with ChatGPT" subscription)
through ``codex exec`` as a subprocess.

We use the CLI rather than the ``openai-codex`` Python SDK because the SDK bundles
a prebuilt `codex` binary that does not build in the uv2nix venv (see
LESSONS.md `codex-binary-breaks-uv2nix-venv`); the nixpkgs `codex` runs fine
on NixOS and shares its auth under ``CODEX_HOME``. Using a ChatGPT subscription
programmatically is a personal-use gray area (tasks/20260719-153040/SPIKE.md), so
the agent is off unless the operator enables it and has run ``codex login``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Literal,
    NamedTuple,
    Protocol,
    runtime_checkable,
)

from pydantic import BaseModel, Field

from .config import Settings
from .logsetup import truncate

# ToolCall/TokenUsage now live in sessions.py (so TranscriptMessage can carry them
# without an import cycle); imported here so they are still used and re-exported as
# scufris.agent.ToolCall / .TokenUsage for existing callers.
from .sessions import STEERING_PREAMBLE, TokenUsage, ToolCall

logger = logging.getLogger(__name__)


class AgentUnavailable(RuntimeError):
    """Raised when the agent cannot serve a request (disabled or unconfigured)."""


class AgentReply(BaseModel):
    text: str
    status: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage | None = None


# Events streamed during one turn (SSE), so the UI can show live progress. A
# `tool` fires as each MCP tool completes; `done` carries the final reply; `error`
# reports a failed turn. The `kind` field discriminates them on the wire.
class StreamTool(BaseModel):
    kind: Literal["tool"] = "tool"
    tool: ToolCall


class StreamDone(BaseModel):
    kind: Literal["done"] = "done"
    reply: AgentReply
    session_id: str | None = None


class StreamError(BaseModel):
    kind: Literal["error"] = "error"
    detail: str


# app-server-only: token-by-token assistant text, and reasoning ("thinking").
class StreamTextDelta(BaseModel):
    kind: Literal["text_delta"] = "text_delta"
    delta: str


class StreamReasoningDelta(BaseModel):
    kind: Literal["reasoning_delta"] = "reasoning_delta"
    delta: str


StreamEvent = (
    StreamTool | StreamDone | StreamError | StreamTextDelta | StreamReasoningDelta
)


@runtime_checkable
class Agent(Protocol):
    """What the chat layer depends on. Implementations are swappable."""

    async def chat(self, prompt: str) -> AgentReply:
        """Run one turn and return the assistant's reply."""
        ...

    def chat_stream(
        self, prompt: str, image_paths: list[str] | None = None
    ) -> AsyncIterator[StreamEvent]:
        """Run one turn, yielding live progress events (tools, then done).

        ``image_paths`` are local image files to attach to the turn (shown to the
        model); None/empty for a text-only turn.
        """
        ...

    def reset(self) -> None:
        """Start a fresh conversation (forget prior context)."""
        ...

    def current_session_id(self) -> str | None:
        """The codex session id being continued, or None for a fresh chat."""
        ...

    def new_session(self) -> None:
        """Drop the current session so the next turn starts a new one."""
        ...

    def switch_session(self, session_id: str) -> None:
        """Continue an existing codex session on the next turn."""
        ...

    async def aclose(self) -> None:
        """Release any resources held by the agent."""
        ...


class DisabledAgent:
    """Stand-in when the agent is off or unconfigured; every call fails clearly."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    async def chat(self, prompt: str) -> AgentReply:
        raise AgentUnavailable(self._reason)

    async def chat_stream(
        self, prompt: str, image_paths: list[str] | None = None
    ) -> AsyncIterator[StreamEvent]:
        yield StreamError(detail=self._reason)

    def reset(self) -> None:
        return None

    def current_session_id(self) -> str | None:
        return None

    def new_session(self) -> None:
        return None

    def switch_session(self, session_id: str) -> None:
        return None

    async def aclose(self) -> None:
        return None


_MOCK_REPLY = (
    "Here is a **mock** reply - no codex required.\n\n"
    "- it streams token by token\n"
    "- it shows a thinking section and a tool call\n\n"
    "```python\nprint('hello from the scufris mock agent')\n```\n"
)
_MOCK_USAGE = TokenUsage(
    input_tokens=1234,
    cached_input_tokens=0,
    output_tokens=64,
    reasoning_output_tokens=8,
)


def _mock_tokens(text: str) -> list[str]:
    """Split into word-ish tokens (each keeps its trailing space) to stream."""
    return re.findall(r"\S+\s*|\s+", text)


class MockAgent:
    """A canned in-process agent that needs no codex login, subprocess or network.

    Selected with ``SCUFRIS_AGENT_BACKEND=mock``. It streams a reasoning
    ("thinking") section, a tool call and token-by-token markdown text with small
    delays, so the whole streaming UI can be exercised and demoed offline. Session
    switching is faked with an in-memory id.
    """

    def __init__(self) -> None:
        self._session_id: str | None = None

    def _ensure_session(self) -> str:
        if self._session_id is None:
            self._session_id = f"mock-{uuid.uuid4()}"
        return self._session_id

    async def chat(self, prompt: str) -> AgentReply:
        self._ensure_session()
        return AgentReply(text=_MOCK_REPLY, status="completed", usage=_MOCK_USAGE)

    async def chat_stream(
        self, prompt: str, image_paths: list[str] | None = None
    ) -> AsyncIterator[StreamEvent]:
        session_id = self._ensure_session()
        seen = f" (+{len(image_paths)} image)" if image_paths else ""
        for chunk in (
            "Reading the request... ",
            f"you said {prompt!r}{seen}. ",
            "Answering.",
        ):
            yield StreamReasoningDelta(delta=chunk)
            await asyncio.sleep(0.02)
        tool = ToolCall(server="scufris", tool="host_stats", status="completed")
        yield StreamTool(tool=tool)
        for token in _mock_tokens(_MOCK_REPLY):
            yield StreamTextDelta(delta=token)
            await asyncio.sleep(0.03)
        yield StreamDone(
            reply=AgentReply(
                text=_MOCK_REPLY,
                status="completed",
                tool_calls=[tool],
                usage=_MOCK_USAGE,
            ),
            session_id=session_id,
        )

    def reset(self) -> None:
        self._session_id = None

    def current_session_id(self) -> str | None:
        return self._session_id

    def new_session(self) -> None:
        self._session_id = None

    def switch_session(self, session_id: str) -> None:
        self._session_id = session_id

    async def aclose(self) -> None:
        return None


def _resolve_codex_bin(settings: Settings) -> str:
    codex_bin = settings.codex_bin or shutil.which("codex")
    if not codex_bin:
        raise AgentUnavailable(
            "codex CLI not found. Install it (nixpkgs `codex`, already in the "
            "dev shell) or set SCUFRIS_CODEX_BIN."
        )
    return codex_bin


def _codex_env(settings: Settings) -> dict[str, str]:
    env = dict(os.environ)
    if settings.codex_home is not None:
        env["CODEX_HOME"] = str(settings.codex_home)
    return env


class TurnOutcome(NamedTuple):
    text: str
    thread_id: str | None
    tool_calls: list[ToolCall] = []
    usage: TokenUsage | None = None


# The runner is the injectable seam: production shells out to `codex exec`; tests
# pass a fake so no codex binary or network is needed. It receives the current
# codex thread id (None for a new conversation) and returns the reply plus the
# thread id to continue from.
CodexRunner = Callable[[Settings, str, "str | None"], Awaitable[TurnOutcome]]


def _parse_event_line(raw: bytes) -> dict[str, Any] | None:
    """Parse one `codex exec --json` line into a dict, or None if malformed."""
    line = raw.strip()
    if not line:
        return None
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _usage_from(raw: dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        input_tokens=int(raw.get("input_tokens") or 0),
        cached_input_tokens=int(raw.get("cached_input_tokens") or 0),
        output_tokens=int(raw.get("output_tokens") or 0),
        reasoning_output_tokens=int(raw.get("reasoning_output_tokens") or 0),
    )


def _tool_call_from(item: dict[str, Any]) -> ToolCall:
    return ToolCall(
        server=str(item.get("server") or ""),
        tool=str(item.get("tool") or ""),
        status=str(item.get("status") or ""),
    )


def _parse_events(
    stdout: bytes,
) -> tuple[str | None, list[ToolCall], TokenUsage | None]:
    """Parse the `codex exec --json` event stream for one turn.

    Recovers the thread id (`thread.started`), the MCP tool calls (each
    `item.completed` of type `mcp_tool_call`), and the token usage
    (`turn.completed`). Malformed lines are skipped.
    """
    thread_id: str | None = None
    tool_calls: list[ToolCall] = []
    usage: TokenUsage | None = None
    for raw in stdout.splitlines():
        event = _parse_event_line(raw)
        if event is None:
            continue
        etype = event.get("type")
        if etype == "thread.started":
            tid = event.get("thread_id")
            if isinstance(tid, str):
                thread_id = tid
        elif etype == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "mcp_tool_call":
                tool_calls.append(_tool_call_from(item))
        elif etype == "turn.completed":
            raw_usage = event.get("usage")
            if isinstance(raw_usage, dict):
                usage = _usage_from(raw_usage)
    return thread_id, tool_calls, usage


# A server id becomes a TOML key (`mcp_servers.<id>.*`), so restrict it to a
# plain identifier - this keeps a config-supplied id from injecting extra keys.
_SERVER_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _server_override(
    server_id: str, command: str, args: list[str], approve: bool
) -> list[str]:
    """The `-c` lines registering one MCP server for a codex-exec invocation."""
    out = ["-c", f"mcp_servers.{server_id}.command={json.dumps(command)}"]
    if args:
        out += ["-c", f"mcp_servers.{server_id}.args={json.dumps(args)}"]
    if approve:
        out += [
            "-c",
            f'mcp_servers.{server_id}.default_tools_approval_mode="approve"',
        ]
    return out


def _mcp_overrides(settings: Settings) -> list[str]:
    """`-c` config registering the MCP servers for this invocation.

    Injected per codex-exec call so nothing is written to `~/.codex`. The built-in
    Scufris server runs with this interpreter (`python -m scufris.mcp_server`);
    any operator-declared `settings.mcp_servers` are appended. For unattended
    `codex exec`, MCP tool calls would otherwise be auto-cancelled (no stdin to
    approve on), so trusted servers auto-approve their tools and approval_policy
    is never. The read-only sandbox (set on turn 1) remains the real guardrail.
    """
    if not settings.agent_tools_enabled:
        return []
    args = _server_override(
        "scufris", sys.executable, ["-m", "scufris.mcp_server"], approve=True
    )
    for spec in settings.mcp_servers:
        # Skip an invalid id or one colliding with the built-in server rather
        # than emitting a malformed / overriding `-c`.
        if spec.id == "scufris" or not _SERVER_ID_RE.match(spec.id):
            continue
        args += _server_override(spec.id, spec.command, spec.args, spec.approve)
    args += ["-c", 'approval_policy="never"']
    return args


def _steer(settings: Settings, prompt: str) -> str:
    """Prepend the tool-steering preamble to a turn's prompt when tools are enabled.

    codex ignores softer channels (tool descriptions, instructions files) and only
    obeys the turn prompt, so the steering rides on the prompt itself; it is
    stripped from titles/transcripts on read (``sessions.strip_steering``). No
    steering when tools are disabled - there is nothing to prefer.
    """
    if not settings.agent_tools_enabled:
        return prompt
    return f"{STEERING_PREAMBLE}\n\n{prompt}"


def _exec_args(
    codex_bin: str,
    settings: Settings,
    prompt: str,
    thread_id: str | None,
    out_file: Path,
    image_paths: list[str] | None = None,
) -> list[str]:
    """Build the `codex exec [resume]` argument list (shared by both runners)."""
    args = [codex_bin, "exec"]
    if thread_id:
        args.append("resume")
    args += ["--json", "--output-last-message", str(out_file)]
    # `resume` inherits the original session's sandbox and rejects the flag; only
    # set it on the first turn (it persists to resumes).
    if not thread_id:
        args += ["--sandbox", "read-only"]
    args += ["--skip-git-repo-check"]
    args += _mcp_overrides(settings)
    if settings.agent_model:
        args += ["--model", settings.agent_model]
    # Attached images: `codex exec -i <FILE>...` shows them to the model.
    for path in image_paths or []:
        args += ["--image", path]
    if thread_id:
        args.append(thread_id)
    args.append(_steer(settings, prompt))
    return args


def _exec_mode(thread_id: str | None) -> str:
    return "resume" if thread_id else "new"


def _log_tool_call(call: ToolCall) -> None:
    logger.info("tool %s.%s -> %s", call.server, call.tool, call.status)


def _log_usage(usage: TokenUsage | None) -> None:
    if usage is not None:
        logger.info(
            "usage input=%d cached=%d output=%d reasoning=%d",
            usage.input_tokens,
            usage.cached_input_tokens,
            usage.output_tokens,
            usage.reasoning_output_tokens,
        )


async def _run_codex_exec(
    settings: Settings, prompt: str, thread_id: str | None = None
) -> TurnOutcome:
    """Run one `codex exec` turn, resuming ``thread_id`` when given.

    Read-only sandbox. ``--json`` recovers the thread id (for continuity) and
    ``--output-last-message`` captures the reply text. Sessions persist (no
    ``--ephemeral``) so a later turn can resume them. The Scufris MCP tools are
    registered per-invocation via ``-c`` (no `~/.codex` edits).
    """
    codex_bin = _resolve_codex_bin(settings)
    mode = _exec_mode(thread_id)
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / "reply.txt"
        args = _exec_args(codex_bin, settings, prompt, thread_id, out_file)
        # Prompt truncated; the API key is never in the argv (auth is codex's).
        logger.debug(
            "codex exec %s model=%s prompt=%r",
            mode,
            settings.agent_model or "(default)",
            truncate(prompt, 160),
        )

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_codex_env(settings),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=settings.agent_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            logger.warning(
                "codex exec %s timed out after %ss",
                mode,
                settings.agent_timeout_seconds,
            )
            raise AgentUnavailable(
                f"codex exec timed out after {settings.agent_timeout_seconds}s"
            ) from exc

        if proc.returncode != 0:
            detail = (
                stderr.decode(errors="replace").strip() or f"exit {proc.returncode}"
            )
            logger.error("codex exec %s failed: %s", mode, truncate(detail, 300))
            raise AgentUnavailable(f"codex exec failed: {detail}")

        parsed_thread_id, tool_calls, usage = _parse_events(stdout)
        new_thread_id = parsed_thread_id or thread_id
        for call in tool_calls:
            _log_tool_call(call)
        _log_usage(usage)
        logger.info(
            "codex exec %s -> ok tools=%d in %.2fs",
            mode,
            len(tool_calls),
            time.monotonic() - started,
        )
        try:
            text = out_file.read_text().strip()
        except OSError:
            text = ""
        return TurnOutcome(
            text=text,
            thread_id=new_thread_id,
            tool_calls=tool_calls,
            usage=usage,
        )


# The streaming seam: yields StreamEvents as codex emits `--json` lines, so the UI
# can show live tool progress. Tests pass a fake async generator.
StreamRunner = Callable[
    [Settings, str, "str | None", "list[str] | None"], AsyncIterator[StreamEvent]
]


async def _stream_codex_exec(
    settings: Settings,
    prompt: str,
    thread_id: str | None = None,
    image_paths: list[str] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Run one `codex exec` turn, yielding events as its `--json` lines arrive.

    Same invocation as `_run_codex_exec`, but stdout is read line-by-line so an
    `mcp_tool_call` `item.completed` becomes a live `StreamTool` the moment the
    tool finishes; the final reply is a `StreamDone`. A stalled turn (no output
    within the timeout) or a nonzero exit becomes a `StreamError`. The subprocess
    is killed in `finally` if the generator is closed early (client disconnect).
    """
    codex_bin = _resolve_codex_bin(settings)
    timeout = settings.agent_timeout_seconds
    mode = _exec_mode(thread_id)
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / "reply.txt"
        args = _exec_args(codex_bin, settings, prompt, thread_id, out_file, image_paths)
        logger.debug(
            "codex exec stream %s model=%s prompt=%r",
            mode,
            settings.agent_model or "(default)",
            truncate(prompt, 160),
        )
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_codex_env(settings),
        )
        assert proc.stdout is not None
        parsed_thread_id: str | None = None
        tool_calls: list[ToolCall] = []
        usage: TokenUsage | None = None
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    proc.kill()
                    logger.warning("codex exec stream %s timed out", mode)
                    yield StreamError(detail=f"codex exec timed out after {timeout}s")
                    return
                try:
                    raw = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("codex exec stream %s timed out", mode)
                    yield StreamError(detail=f"codex exec timed out after {timeout}s")
                    return
                if not raw:
                    break
                event = _parse_event_line(raw)
                if event is None:
                    continue
                # Every raw event line at DEBUG - the deepest trace of a turn.
                logger.debug(
                    "codex json: %s",
                    truncate(raw.decode(errors="replace").strip(), 200),
                )
                etype = event.get("type")
                if etype == "thread.started":
                    tid = event.get("thread_id")
                    if isinstance(tid, str):
                        parsed_thread_id = tid
                elif etype == "item.completed":
                    item = event.get("item")
                    if isinstance(item, dict) and item.get("type") == "mcp_tool_call":
                        call = _tool_call_from(item)
                        tool_calls.append(call)
                        _log_tool_call(call)
                        yield StreamTool(tool=call)
                elif etype == "turn.completed":
                    raw_usage = event.get("usage")
                    if isinstance(raw_usage, dict):
                        usage = _usage_from(raw_usage)
                        _log_usage(usage)

            await proc.wait()
            if proc.returncode not in (0, None):
                stderr = await proc.stderr.read() if proc.stderr else b""
                detail = stderr.decode(errors="replace").strip() or (
                    f"exit {proc.returncode}"
                )
                logger.error(
                    "codex exec stream %s failed: %s", mode, truncate(detail, 300)
                )
                yield StreamError(detail=f"codex exec failed: {detail}")
                return
            logger.info(
                "codex exec stream %s -> ok tools=%d in %.2fs",
                mode,
                len(tool_calls),
                time.monotonic() - started,
            )
            try:
                text = out_file.read_text().strip()
            except OSError:
                text = ""
            reply = AgentReply(
                text=text,
                status="completed",
                tool_calls=tool_calls,
                usage=usage,
            )
            yield StreamDone(reply=reply, session_id=parsed_thread_id or thread_id)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()


# --- codex app-server (experimental JSON-RPC) streaming backend ----------------
#
# Unlike `codex exec` (turn-level), the app-server streams `item/agentMessage/delta`
# (token-by-token text) and `item/reasoning/textDelta` ("thinking"). We drive it
# over newline-delimited JSON-RPC on stdio: initialize -> thread/start (or
# thread/resume) -> turn/start, then read notifications until turn/completed.


def _appserver_event(obj: dict[str, Any]) -> StreamEvent | None:
    """Map one app-server notification to a StreamEvent (or None to skip)."""
    method = obj.get("method")
    params = obj.get("params")
    if not isinstance(params, dict):
        return None
    if method == "item/agentMessage/delta":
        delta = params.get("delta")
        return StreamTextDelta(delta=delta) if isinstance(delta, str) else None
    if method in ("item/reasoning/textDelta", "item/reasoning/summaryTextDelta"):
        delta = params.get("delta")
        return StreamReasoningDelta(delta=delta) if isinstance(delta, str) else None
    if method == "item/completed":
        item = params.get("item")
        if isinstance(item, dict) and "tool" in str(item.get("type", "")).lower():
            return StreamTool(
                tool=ToolCall(
                    server=str(item.get("server") or "scufris"),
                    tool=str(item.get("tool") or item.get("name") or item.get("type")),
                    status=str(item.get("status") or "completed"),
                )
            )
    return None


def _appserver_usage(params: dict[str, Any]) -> TokenUsage | None:
    total = params.get("tokenUsage")
    total = total.get("total") if isinstance(total, dict) else None
    if not isinstance(total, dict):
        return None
    return TokenUsage(
        input_tokens=int(total.get("inputTokens") or 0),
        cached_input_tokens=int(total.get("cachedInputTokens") or 0),
        output_tokens=int(total.get("outputTokens") or 0),
        reasoning_output_tokens=int(total.get("reasoningOutputTokens") or 0),
    )


async def _appserver_call(
    proc: asyncio.subprocess.Process,
    request_id: int,
    method: str,
    params: dict[str, Any],
    deadline: float,
) -> dict[str, Any]:
    """Send a JSON-RPC request and read (ignoring notifications) until its reply."""
    assert proc.stdin is not None and proc.stdout is not None
    payload = json.dumps({"id": request_id, "method": method, "params": params})
    proc.stdin.write((payload + "\n").encode())
    await proc.stdin.drain()
    loop = asyncio.get_event_loop()
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(f"app-server {method} timed out")
        raw = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
        if not raw:
            raise AgentUnavailable(f"app-server closed during {method}")
        message = _parse_event_line(raw)
        if message and message.get("id") == request_id:
            return message


async def _stream_app_server(
    settings: Settings,
    prompt: str,
    thread_id: str | None = None,
    image_paths: list[str] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream one turn via `codex app-server`, yielding token/reasoning/tool events."""
    codex_bin = _resolve_codex_bin(settings)
    timeout = settings.agent_timeout_seconds
    mode = _exec_mode(thread_id)
    started = time.monotonic()
    args = [codex_bin, "app-server", *_mcp_overrides(settings)]
    logger.debug(
        "app-server %s model=%s prompt=%r",
        mode,
        settings.agent_model or "(default)",
        truncate(prompt, 160),
    )
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_codex_env(settings),
    )
    assert proc.stdout is not None and proc.stdin is not None
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    rid = 0
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    usage: TokenUsage | None = None
    new_thread_id = thread_id
    try:
        rid += 1
        await _appserver_call(
            proc,
            rid,
            "initialize",
            {
                "clientInfo": {"name": "scufris", "title": None, "version": "0"},
                "capabilities": None,
            },
            deadline,
        )
        rid += 1
        if thread_id:
            resp = await _appserver_call(
                proc, rid, "thread/resume", {"threadId": thread_id}, deadline
            )
        else:
            start_params: dict[str, Any] = {"sandbox": "read-only"}
            if settings.agent_model:
                start_params["model"] = settings.agent_model
            resp = await _appserver_call(
                proc, rid, "thread/start", start_params, deadline
            )
        result = resp.get("result")
        if not isinstance(result, dict) or "error" in resp:
            detail = json.dumps(resp.get("error") or resp)[:300]
            logger.error("app-server thread setup failed: %s", detail)
            yield StreamError(detail=f"app-server thread setup failed: {detail}")
            return
        thread = result.get("thread")
        if isinstance(thread, dict) and isinstance(thread.get("id"), str):
            new_thread_id = thread["id"]

        rid += 1
        # The turn input is an array of UserInput items; attached images ride as
        # `localImage` items (a local file path) alongside the text.
        turn_input: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": _steer(settings, prompt),
                "text_elements": [],
            }
        ]
        for path in image_paths or []:
            turn_input.append({"type": "localImage", "path": path})
        await _appserver_call(
            proc,
            rid,
            "turn/start",
            {
                "threadId": new_thread_id,
                "input": turn_input,
            },
            deadline,
        )

        # The turn streams as notifications until turn/completed.
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                proc.kill()
                logger.warning("app-server %s timed out", mode)
                yield StreamError(detail=f"app-server timed out after {timeout}s")
                return
            raw = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
            if not raw:
                break
            message = _parse_event_line(raw)
            if message is None or "id" in message:
                continue  # skip responses/malformed
            method = message.get("method")
            params = message.get("params")
            if method == "thread/tokenUsage/updated" and isinstance(params, dict):
                usage = _appserver_usage(params) or usage
            event = _appserver_event(message)
            if event is not None:
                if isinstance(event, StreamTextDelta):
                    text_parts.append(event.delta)
                elif isinstance(event, StreamTool):
                    tool_calls.append(event.tool)
                    _log_tool_call(event.tool)
                yield event
            if method == "turn/completed":
                break

        logger.info(
            "app-server %s -> ok tools=%d in %.2fs",
            mode,
            len(tool_calls),
            time.monotonic() - started,
        )
        _log_usage(usage)
        reply = AgentReply(
            text="".join(text_parts).strip(),
            status="completed",
            tool_calls=tool_calls,
            usage=usage,
        )
        yield StreamDone(reply=reply, session_id=new_thread_id)
    except (TimeoutError, asyncio.TimeoutError):
        proc.kill()
        logger.warning("app-server %s timed out", mode)
        yield StreamError(detail=f"app-server timed out after {timeout}s")
    except AgentUnavailable as exc:
        yield StreamError(detail=str(exc))
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


class CodexCliAgent:
    """Drive Codex via `codex exec`, keeping one conversation across turns.

    The codex thread id is remembered so turns share context; ``reset`` starts a
    fresh conversation.
    """

    def __init__(
        self,
        settings: Settings,
        runner: CodexRunner = _run_codex_exec,
        stream_runner: StreamRunner = _stream_codex_exec,
    ) -> None:
        self._settings = settings
        self._runner = runner
        self._stream_runner = stream_runner
        # The codex session (a.k.a. thread) id currently being continued. None
        # means the next turn opens a fresh session; switching sets it to an
        # existing id to resume that conversation.
        self._session_id: str | None = None

    async def chat(self, prompt: str) -> AgentReply:
        outcome = await self._runner(self._settings, prompt, self._session_id)
        self._session_id = outcome.thread_id
        return AgentReply(
            text=outcome.text,
            status="completed",
            tool_calls=outcome.tool_calls,
            usage=outcome.usage,
        )

    async def chat_stream(
        self, prompt: str, image_paths: list[str] | None = None
    ) -> AsyncIterator[StreamEvent]:
        async for event in self._stream_runner(
            self._settings, prompt, self._session_id, image_paths
        ):
            if isinstance(event, StreamDone):
                self._session_id = event.session_id
            yield event

    def reset(self) -> None:
        self.new_session()

    def current_session_id(self) -> str | None:
        return self._session_id

    def new_session(self) -> None:
        self._session_id = None

    def switch_session(self, session_id: str) -> None:
        self._session_id = session_id

    async def aclose(self) -> None:
        return None


def build_agent(
    settings: Settings,
    runner: CodexRunner = _run_codex_exec,
    stream_runner: StreamRunner | None = None,
) -> Agent:
    """Select the agent implementation from settings.

    The backend follows ``settings.agent_backend`` unless a ``stream_runner`` is
    injected (tests): ``app_server`` streams token-by-token, ``exec`` is the
    turn-level path, ``mock`` is a canned offline agent. Non-streaming ``chat``
    uses exec (or the mock).
    """
    if not settings.agent_enabled:
        return DisabledAgent(
            "agent is disabled. Set SCUFRIS_AGENT_ENABLED=1 and run `codex login` "
            "(or `scufris login`) to enable it."
        )
    if settings.agent_backend == "mock":
        return MockAgent()
    if stream_runner is None:
        stream_runner = (
            _stream_app_server
            if settings.agent_backend == "app_server"
            else _stream_codex_exec
        )
    return CodexCliAgent(settings, runner=runner, stream_runner=stream_runner)


async def login(settings: Settings, *, printer: Callable[[str], None] = print) -> None:
    """Authenticate Codex for this host by delegating to `codex login`.

    In chatgpt mode this runs the interactive browser/device flow (stdio is
    inherited). In api_key mode the key is piped to ``codex login --with-api-key``.
    """
    codex_bin = _resolve_codex_bin(settings)
    env = _codex_env(settings)

    if settings.agent_auth_mode == "api_key":
        if not settings.openai_api_key:
            raise AgentUnavailable(
                "agent_auth_mode=api_key but SCUFRIS_OPENAI_API_KEY is unset."
            )
        printer("Logging in with API key via `codex login --with-api-key`...")
        proc = await asyncio.create_subprocess_exec(
            codex_bin,
            "login",
            "--with-api-key",
            stdin=asyncio.subprocess.PIPE,
            env=env,
        )
        await proc.communicate(settings.openai_api_key.encode())
    else:
        printer("Launching `codex login` (Sign in with ChatGPT)...")
        proc = await asyncio.create_subprocess_exec(codex_bin, "login", env=env)
        await proc.wait()

    if proc.returncode != 0:
        raise AgentUnavailable(f"codex login exited with status {proc.returncode}")
    printer("Codex login complete.")
