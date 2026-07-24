"""The Scufris agent backend: the codex app-server runner and shared agent types.

This module holds the codex ``app-server`` streaming runner, the streaming event
types, and ``login`` - the low-level plumbing that the swappable ``AgentBackend``
implementations in ``backends.py`` drive (the old ``Agent`` protocol and its
``CodexCliAgent``/``AgentHandle``/``MockAgent`` classes were retired in B5bc, and
the turn-level ``codex exec`` runners in B5e; the orchestrator and every agent now
run through ``get_backend(...).stream()``, which streams token-by-token via the
app-server). The default codex path drives the ``codex`` CLI (nixpkgs `codex`,
"Sign in with ChatGPT" subscription) through a ``codex app-server`` subprocess.

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
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Literal,
)

from pydantic import BaseModel, Field

from .config import SERVER_ID_RE, Settings
from .logsetup import truncate

# ToolCall/TokenUsage now live in sessions.py (so TranscriptMessage can carry them
# without an import cycle); imported here so they are still used and re-exported as
# scufris.agent.ToolCall / .TokenUsage for existing callers.
from .sessions import AGENT_STEERING_PREAMBLE, STEERING_PREAMBLE, TokenUsage, ToolCall

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


# A server id becomes a TOML key (`mcp_servers.<id>.*`), so restrict it to a
# plain identifier - this keeps a config-supplied id from injecting extra keys.
# Compiled from the single source in config.py so the two id boundaries (this
# skip + the settings endpoint) can never drift.
_SERVER_ID_RE = re.compile(SERVER_ID_RE)


def _server_override(
    server_id: str,
    command: str,
    args: list[str],
    approve: bool,
    env: dict[str, str] | None = None,
) -> list[str]:
    """The `-c` lines registering one MCP server for a codex invocation."""
    out = ["-c", f"mcp_servers.{server_id}.command={json.dumps(command)}"]
    if args:
        out += ["-c", f"mcp_servers.{server_id}.args={json.dumps(args)}"]
    if approve:
        out += [
            "-c",
            f'mcp_servers.{server_id}.default_tools_approval_mode="approve"',
        ]
    for key, value in (env or {}).items():
        out += ["-c", f"mcp_servers.{server_id}.env.{key}={json.dumps(value)}"]
    return out


@dataclass(frozen=True)
class ScufrisMcpServer:
    """The backend-agnostic registration of the built-in scufris MCP server for one
    turn: the process to launch and the ROLE-SCOPED env that picks its audience.

    This is the single source of truth every backend formats to its own flavour -
    codex to ``-c mcp_servers.scufris.*`` overrides (``_mcp_overrides``), claude to a
    ``--mcp-config`` JSON blob (``backends._scufris_claude_args``). The CONTENT
    (command, args, the ``SCUFRIS_AGENT_ROLE`` / ``SCUFRIS_AGENT_ID`` /
    ``SCUFRIS_DISABLED_TOOLS`` env) is identical across backends; only the FORMAT
    differs, so the two can never drift on what a role exposes. The per-role tool
    surface is enforced by the SERVER (``mcp_server.apply_role`` reads
    ``SCUFRIS_AGENT_ROLE``), so a backend only has to allow-list the whole scufris
    server, not enumerate the role's tool names.
    """

    command: str
    args: tuple[str, ...]
    env: dict[str, str]


def scufris_mcp_server(
    settings: Settings, *, is_orchestrator: bool = False, agent_id: str = ""
) -> ScufrisMcpServer | None:
    """The scufris MCP server registration for this turn, or ``None`` when the turn
    gets no scufris server: tools disabled, or a regular agent turn with no id
    (nothing to address the ``request_input`` callback back to).

    Role selection mirrors ``mcp_server.ROLE_ORCHESTRATOR`` / ``ROLE_AGENT``: the
    orchestrator gets the full host/observe/control surface plus the operator's
    disabled-tool set and the dashboard API base for its control tools; a regular
    agent gets the ``agent`` role - ONLY the ``request_input`` callback - plus its
    own id so that callback can POST back. ``is_orchestrator`` wins over
    ``agent_id`` (the landing orchestrator is never a regular agent).
    """
    if not settings.agent_tools_enabled:
        return None
    api_base = f"http://{settings.host}:{settings.port}"
    if is_orchestrator:
        env: dict[str, str] = {
            "SCUFRIS_API_BASE": api_base,
            "SCUFRIS_AGENT_ROLE": "orchestrator",
        }
        if settings.disabled_tools:
            env["SCUFRIS_DISABLED_TOOLS"] = ",".join(settings.disabled_tools)
    elif agent_id:
        env = {
            "SCUFRIS_API_BASE": api_base,
            "SCUFRIS_AGENT_ROLE": "agent",
            "SCUFRIS_AGENT_ID": agent_id,
        }
    else:
        return None
    return ScufrisMcpServer(
        command=sys.executable,
        args=("-m", "scufris.mcp_server"),
        env=env,
    )


def _mcp_overrides(
    settings: Settings, *, is_orchestrator: bool = False, agent_id: str = ""
) -> list[str]:
    """`-c` config registering the MCP servers for this invocation.

    Injected on the `codex app-server` argv so nothing is written to `~/.codex`.
    The built-in Scufris server (`python -m scufris.mcp_server`) is ROLE-SCOPED
    (values mirror ``mcp_server.ROLE_ORCHESTRATOR`` / ``ROLE_AGENT``): the
    orchestrator gets the full host/observe/control surface; a regular agent gets
    the ``agent`` role - ONLY the ``request_input`` callback - plus its own id so
    that callback can address itself. The server enforces the scoping at startup
    (``mcp_server.apply_role``); this only picks the role via env. Regular agents
    get no other scufris tools and draw the rest from their project config/skills.
    Any operator-declared ``settings.mcp_servers`` are global config and are
    appended for EVERY agent. For an unattended codex run, MCP tool calls would
    otherwise be auto-cancelled (no stdin to approve on), so trusted servers
    auto-approve their tools and approval_policy is never. The sandbox (set per
    turn on thread/start|resume) remains the real guardrail.
    """
    if not settings.agent_tools_enabled:
        return []
    args: list[str] = []
    # The role-scoped scufris server (orchestrator: full surface + disabled-tool
    # set + API base; agent: only request_input + its own id) comes from the shared
    # core so codex and claude never drift on it; codex formats it to `-c` overrides
    # here, auto-approving the whole server since an unattended run has no stdin to
    # approve on.
    server = scufris_mcp_server(
        settings, is_orchestrator=is_orchestrator, agent_id=agent_id
    )
    if server is not None:
        args += _server_override(
            "scufris",
            server.command,
            list(server.args),
            approve=True,
            env=server.env,
        )
    for spec in settings.mcp_servers:
        # Skip an invalid id or one colliding with the built-in server rather
        # than emitting a malformed / overriding `-c`.
        if spec.id == "scufris" or not _SERVER_ID_RE.fullmatch(spec.id):
            continue
        args += _server_override(spec.id, spec.command, spec.args, spec.approve)
    args += ["-c", 'approval_policy="never"']
    return args


def _steer(
    settings: Settings,
    prompt: str,
    *,
    is_orchestrator: bool = False,
    agent_id: str = "",
) -> str:
    """Prepend the role's tool-steering preamble to a turn's prompt when tools are on.

    codex ignores softer channels (tool descriptions, instructions files) and only
    obeys the turn prompt, so the steering rides on the prompt itself; it is
    stripped from titles/transcripts on read (``sessions.strip_steering``). This is
    the CODEX turn path (``_stream_app_server``); the claude backend wires the same
    role-scoped scufris server but does not run this preamble (claude honours the
    softer channels), so a claude turn is unsteered by this function regardless. The
    role picks the preamble, mirroring which scufris server ``scufris_mcp_server``
    grants:

    - the orchestrator (host-tools server) gets ``STEERING_PREAMBLE``, pointing at
      host_stats / disk_usage / list_processes;
    - a sub-agent that ACTUALLY holds the ``request_input`` callback (the agent-role
      server: ``agent_id`` set) gets ``AGENT_STEERING_PREAMBLE``, telling it to
      signal when blocked;
    - any other turn - one with no role or a tools-disabled turn - is left unsteered.
    """
    if not settings.agent_tools_enabled:
        return prompt
    if is_orchestrator:
        return f"{STEERING_PREAMBLE}\n\n{prompt}"
    if agent_id:
        return f"{AGENT_STEERING_PREAMBLE}\n\n{prompt}"
    return prompt


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
        raw = await asyncio.wait_for(proc.stdout.readline(), timeout=idle)
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

    ``is_orchestrator`` selects the orchestrator role of the scufris MCP server and
    its tool-steering preamble (see ``_mcp_overrides`` / ``_steer``); a regular
    agent turn passes False, gets the agent role of the scufris server (only the
    ``request_input`` callback), and no steering. ``agent_id`` is the caller's own
    id, threaded to a regular agent's scufris server so ``request_input`` can
    address itself back to the API.
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
        *_mcp_overrides(settings, is_orchestrator=is_orchestrator, agent_id=agent_id),
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
            # turn (task 20260721-183828). ThreadResumeParams accepts `sandbox`.
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
            raw = await asyncio.wait_for(proc.stdout.readline(), timeout=idle)
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
