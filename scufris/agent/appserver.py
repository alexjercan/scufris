"""The `codex app-server` turn: steering, JSON-RPC plumbing, and the event stream.

Unlike `codex exec` (turn-level), the app-server streams `item/agentMessage/delta`
(token-by-token text) and `item/reasoning/textDelta` ("thinking"). We drive it
over newline-delimited JSON-RPC on stdio: initialize -> thread/start (or
thread/resume) -> turn/start, then read notifications until turn/completed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from typing import Any, AsyncIterator

from ..config import Settings
from ..enums import Audience, audience_for
from ..logsetup import truncate
from ..sessions import (
    AGENT_STEERING_PREAMBLE,
    HOST_STEERING_PREAMBLE,
    STEERING_PREAMBLE,
    TokenUsage,
    ToolCall,
)
from .env import _codex_env, _resolve_codex_bin
from .events import (
    STREAM_READ_LIMIT,
    AgentReply,
    AgentUnavailable,
    StreamDone,
    StreamError,
    StreamEvent,
    StreamReasoningDelta,
    StreamSessionStarted,
    StreamTextDelta,
    StreamTool,
)
from .mcp import _mcp_overrides

logger = logging.getLogger(__name__)


def _parse_event_line(raw: bytes) -> dict[str, Any] | None:
    """Parse one JSON-RPC / `--json` line into a dict, or None if malformed."""
    line = raw.strip()
    if not line:
        return None
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _steer(
    settings: Settings,
    prompt: str,
    *,
    is_orchestrator: bool = False,
    agent_id: str = "",
) -> str:
    """Prepend the audience's tool-steering preamble to a turn's prompt when tools are on.

    codex ignores softer channels (tool descriptions, instructions files) and only
    obeys the turn prompt, so the steering rides on the prompt itself; it is
    stripped from titles/transcripts on read (``sessions.strip_steering``). This is
    the CODEX turn path (``_stream_app_server``); the claude backend wires the same
    scufris servers but does not run this preamble (claude honours the softer
    channels), so a claude turn is unsteered by this function regardless. The
    audience picks the preamble, mirroring which scufris servers
    ``scufris_mcp_servers`` grants:

    - the orchestrator (``scufris`` + ``den`` servers) gets ``STEERING_PREAMBLE``,
      pointing at the read-only host tools and at delegating a host CHANGE to the
      host agent (it has no propose tool to reach for);
    - the HOST agent (the ``host`` + ``agent`` servers) gets
      ``HOST_STEERING_PREAMBLE``, stating the propose/preview/approve contract as
      its normal way of working;
    - a sub-agent that ACTUALLY holds the callbacks (the ``agent`` server:
      ``agent_id`` set) gets ``AGENT_STEERING_PREAMBLE``, telling it to signal when
      blocked;
    - any other turn - one with no audience or a tools-disabled turn - is left
      unsteered.
    """
    if not settings.agent_tools_enabled:
        return prompt
    audience = audience_for(is_orchestrator=is_orchestrator, agent_id=agent_id)
    preamble = {
        Audience.ORCHESTRATOR: STEERING_PREAMBLE,
        Audience.HOST: HOST_STEERING_PREAMBLE,
        Audience.AGENT: AGENT_STEERING_PREAMBLE,
    }.get(audience)
    return f"{preamble}\n\n{prompt}" if preamble is not None else prompt


def _turn_mode(thread_id: str | None) -> str:
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
    idle: float,
) -> dict[str, Any]:
    """Send a JSON-RPC request and read (ignoring notifications) until its reply.

    ``idle`` bounds each individual read, not the whole call: a read that yields
    no line within ``idle`` seconds raises ``asyncio.TimeoutError`` (a hung
    handshake), but a setup that keeps producing lines is never capped by total
    duration - the same no-output semantics the streaming loop uses.
    """
    assert proc.stdin is not None and proc.stdout is not None
    payload = json.dumps({"id": request_id, "method": method, "params": params})
    proc.stdin.write((payload + "\n").encode())
    await proc.stdin.drain()
    while True:
        try:
            raw = await asyncio.wait_for(proc.stdout.readline(), timeout=idle)
        except ValueError as exc:
            # An over-STREAM_READ_LIMIT line during the handshake: re-raise as
            # AgentUnavailable so `_stream_app_server` maps it to a diagnosable
            # StreamError (this call yields nothing itself).
            raise AgentUnavailable(
                f"app-server line exceeded {STREAM_READ_LIMIT}-byte read limit "
                f"during {method}"
            ) from exc
        if not raw:
            raise AgentUnavailable(f"app-server closed during {method}")
        message = _parse_event_line(raw)
        if message and message.get("id") == request_id:
            return message


def _git_writable_roots(cwd: str | None) -> list[str]:
    """Absolute git-metadata dirs `workspace-write` must treat as writable so an
    `edit`-mode agent can still commit.

    codex's `workspace-write` sandbox makes the workspace writable but carves the
    `.git` directory back out as READ-ONLY, to stop the model corrupting history.
    That also blocks the scufris flow (tatr commits, sprout, land), which an
    `edit` agent must be able to run without the full-machine access of `auto`.
    We re-grant exactly the git dirs: the worktree's own git dir AND the shared
    common dir - for a sprout worktree these differ (the common dir is the parent
    repo's `.git`, and a commit writes both), while a plain repo returns one path.
    Returns [] when cwd is absent or not a git repo (nothing to re-grant).
    """
    if not cwd:
        return []
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                cwd,
                "rev-parse",
                "--path-format=absolute",
                "--git-dir",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    roots: list[str] = []
    for line in proc.stdout.splitlines():
        path = line.strip()
        if path and path not in roots:
            roots.append(path)
    return roots


def _sandbox_overrides(sandbox: str, cwd: str | None) -> list[str]:
    """`-c` config re-granting git-dir writes under the `workspace-write` sandbox.

    Only `workspace-write` (scufris `edit` mode) protects `.git`; `read-only`
    (`manual`) ignores writable_roots and `danger-full-access` (`auto`) already
    has full access, so this is a no-op for those. See ``_git_writable_roots``.
    The value is a TOML array of absolute paths (json.dumps emits valid inline
    TOML for a list of strings), appended to the app-server argv so nothing is
    written to ``~/.codex``.
    """
    if sandbox != "workspace-write":
        return []
    roots = _git_writable_roots(cwd)
    if not roots:
        return []
    return ["-c", f"sandbox_workspace_write.writable_roots={json.dumps(roots)}"]


async def _stream_app_server(
    settings: Settings,
    prompt: str,
    thread_id: str | None = None,
    image_paths: list[str] | None = None,
    *,
    cwd: str | None = None,
    sandbox: str = "read-only",
    is_orchestrator: bool = False,
    agent_id: str = "",
) -> AsyncIterator[StreamEvent]:
    """Stream one turn via `codex app-server`, yielding token/reasoning/tool events.

    ``is_orchestrator`` selects the orchestrator's scufris + den servers and its
    tool-steering preamble (see ``_mcp_overrides`` / ``_steer``); a regular agent
    turn passes False and gets ONLY the ``agent`` callback server (the
    ``request_input`` + ``report_back`` callbacks) and no steering - the audience
    split is physical, not a per-server role. ``agent_id`` is the caller's own id,
    threaded to that callback server so the callbacks can address it back to the
    API.
    """
    codex_bin = _resolve_codex_bin(settings)
    # Idle (no-output) bound, NOT a per-turn wall-clock: it caps the gap between
    # app-server lines, so a turn that keeps streaming runs to completion however
    # long it takes, while a genuinely hung app-server is still cut. See the
    # config docstring and ADR-001 (supervisor.py); reset implicitly by reading a
    # fresh `wait_for(timeout=idle)` per line.
    idle = settings.agent_timeout_seconds
    mode = _turn_mode(thread_id)
    started = time.monotonic()
    args = [
        codex_bin,
        "app-server",
        *_mcp_overrides(
            settings,
            is_orchestrator=is_orchestrator,
            agent_id=agent_id,
            # The orchestrator's current chat = the session this turn resumes
            # (empty on a fresh turn); lets a spawned child be stamped with it.
            orch_session_id=thread_id if (is_orchestrator and thread_id) else "",
        ),
        # Re-grant `.git` writes for an `edit` (workspace-write) agent so it can
        # commit; no-op for manual/auto. codex protects `.git` in workspace-write,
        # which would otherwise break the tatr/sprout/land flow (`.git/index.lock:
        # Read-only file system`). See `_sandbox_overrides`.
        *_sandbox_overrides(sandbox, cwd),
    ]
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
        cwd=cwd,
        # A single app-server frame (a big command-output notification) can far
        # exceed asyncio's default 64 KiB readline limit; raise it so such lines
        # stream through instead of raising `ValueError`. See STREAM_READ_LIMIT.
        limit=STREAM_READ_LIMIT,
    )
    assert proc.stdout is not None and proc.stdin is not None
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
            idle,
        )
        rid += 1
        if thread_id:
            # thread/resume MUST re-send the sandbox: each turn spawns a fresh
            # `codex app-server` process, and a resumed thread does NOT restore
            # the sandbox it was started with - it reverts to the default
            # (read-only). Without this, only turn 1 (thread/start) honoured the
            # agent's permission mode; every resumed turn ran read-only, so an
            # `auto`/`edit` agent could not write or run commands after its first
            # turn. ThreadResumeParams accepts `sandbox`.
            resp = await _appserver_call(
                proc,
                rid,
                "thread/resume",
                {"threadId": thread_id, "sandbox": sandbox},
                idle,
            )
        else:
            start_params: dict[str, Any] = {"sandbox": sandbox}
            if settings.agent_model:
                start_params["model"] = settings.agent_model
            resp = await _appserver_call(proc, rid, "thread/start", start_params, idle)
        result = resp.get("result")
        if not isinstance(result, dict) or "error" in resp:
            detail = json.dumps(resp.get("error") or resp)[:300]
            logger.error("app-server thread setup failed: %s", detail)
            yield StreamError(detail=f"app-server thread setup failed: {detail}")
            return
        thread = result.get("thread")
        if isinstance(thread, dict) and isinstance(thread.get("id"), str):
            new_thread_id = thread["id"]
        # Surface the session id as soon as it is known (before the turn streams),
        # so the run-launch path records ownership at turn-start and a mid-turn
        # reattach can find the session. Both thread/start (fresh) and thread/resume
        # populate new_thread_id here.
        if new_thread_id:
            yield StreamSessionStarted(session_id=new_thread_id)

        rid += 1
        # The turn input is an array of UserInput items; attached images ride as
        # `localImage` items (a local file path) alongside the text.
        turn_input: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": _steer(
                    settings,
                    prompt,
                    is_orchestrator=is_orchestrator,
                    agent_id=agent_id,
                ),
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
            idle,
        )

        # The turn streams as notifications until turn/completed. Each read is
        # bounded by the idle guard, not a shared deadline: a turn that keeps
        # emitting lines never times out on total duration, only on silence
        # (readline yielding nothing for `idle`s raises, caught below).
        while True:
            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), timeout=idle)
            except ValueError:
                # readline raises a bare ValueError when a single line overflows
                # STREAM_READ_LIMIT. Surface it as a clean, diagnosable StreamError
                # (the codebase's terminal-error event) instead of letting a raw
                # ValueError propagate to the supervisor as an opaque failure.
                proc.kill()
                logger.warning(
                    "app-server %s line exceeded %d-byte read limit",
                    mode,
                    STREAM_READ_LIMIT,
                )
                yield StreamError(
                    detail=f"app-server line exceeded {STREAM_READ_LIMIT}-byte read limit"
                )
                return
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
        # No app-server line for `idle`s: a stalled handshake or a hung turn.
        proc.kill()
        logger.warning("app-server %s idle-timed out (no output for %ss)", mode, idle)
        yield StreamError(detail=f"app-server timed out after {idle}s")
    except AgentUnavailable as exc:
        yield StreamError(detail=str(exc))
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
