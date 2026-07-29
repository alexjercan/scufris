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

from .auth import API_TOKEN_ENV
from .config import SECRET_ENV_VARS, Settings
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


# Emitted the moment a turn's session (codex thread) id is known - right after
# thread/start|resume, before the turn streams - so a client reattaching mid-turn,
# and the run-launch path, learn the session id without waiting for `done`. codex
# only; other backends carry their id on `done`. See `_stream_app_server`.
class StreamSessionStarted(BaseModel):
    kind: Literal["session_started"] = "session_started"
    session_id: str


# app-server-only: token-by-token assistant text, and reasoning ("thinking").
class StreamTextDelta(BaseModel):
    kind: Literal["text_delta"] = "text_delta"
    delta: str


class StreamReasoningDelta(BaseModel):
    kind: Literal["reasoning_delta"] = "reasoning_delta"
    delta: str


StreamEvent = (
    StreamTool
    | StreamDone
    | StreamError
    | StreamTextDelta
    | StreamReasoningDelta
    | StreamSessionStarted
)


# Max size (bytes) of a single line the JSON-RPC / stream-json readers accept from
# a backend subprocess. asyncio's StreamReader defaults to 64 KiB and raises a bare
# `ValueError` ("Separator is not found, and chunk exceed the limit") on any longer
# line - which for a codex/claude app-server frame is a real, benign occurrence: a
# single command-output notification (a big `rg`, a `tatr ls` over hundreds of
# tasks, a large file dump) easily exceeds 64 KiB. We raise the ceiling to 8 MiB so
# such frames stream through instead of erroring the run. Shared by both the codex
# app-server launch (`_stream_app_server`) and `ClaudeBackend.stream`.
STREAM_READ_LIMIT = 8 * 1024 * 1024


def _resolve_codex_bin(settings: Settings) -> str:
    codex_bin = settings.codex_bin or shutil.which("codex")
    if not codex_bin:
        raise AgentUnavailable(
            "codex CLI not found. Install it (nixpkgs `codex`, already in the "
            "dev shell) or set SCUFRIS_CODEX_BIN."
        )
    return codex_bin


def agent_subprocess_env(settings: Settings) -> dict[str, str]:
    """The environment for EVERY agent child process. The one place it is built.

    Every scufris credential is stripped, because everything the model runs
    inherits this environment - every shell command and every sub-agent.

    This is NOT belt and braces for all of them. The machine API token is minted
    in-process and never put in os.environ (20260729-125015 review round 1,
    finding 2), so stripping it guards against a stale shell. The hostd secret is
    the opposite: it ARRIVES through the environment, because that is how a sops
    secret reaches the unit, so without this the model holds the credential for
    the root helper's socket and can apply host actions with no operator approval
    at all (20260729-125029 review round 1, R1.3). See config.SECRET_ENV_VARS.

    It is a SEAM rather than a call-site strip because the call-site version was
    already forgotten once: the fix for R1.3 stripped codex's environment and the
    claude backend went on spawning with no ``env=`` at all (review round 2,
    R2.1). ``test_no_agent_subprocess_is_spawned_without_the_stripped_environment``
    fails on any agent spawn that does not pass this, so a backend added later is
    covered by the test rather than by someone remembering.
    """
    env = dict(os.environ)
    for name in SECRET_ENV_VARS:
        env.pop(name, None)
    return env


def _codex_env(settings: Settings) -> dict[str, str]:
    """``agent_subprocess_env`` plus codex's own home override."""
    env = agent_subprocess_env(settings)
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
    """One backend-agnostic scufris MCP server registration for a turn: its id, the
    process to launch, and the env that configures it.

    A turn can register SEVERAL of these (an orchestrator turn gets ``scufris`` +
    ``den``; a sub-agent turn gets only ``agent``). Each backend formats them to
    its own flavour - codex to ``-c mcp_servers.<id>.*`` overrides
    (``_mcp_overrides``), claude to a ``--mcp-config`` JSON blob
    (``backends._scufris_claude_args``) - from this ONE source, so the two can
    never drift on which servers/env a turn exposes. The audience split is
    PHYSICAL (which servers are on the turn), not a per-server role filter, so a
    backend only allow-lists each registered server whole.
    """

    server_id: str
    command: str
    args: tuple[str, ...]
    env: dict[str, str]


def scufris_mcp_servers(
    settings: Settings,
    *,
    is_orchestrator: bool = False,
    agent_id: str = "",
    orch_session_id: str = "",
) -> list[ScufrisMcpServer]:
    """The scufris MCP servers to register for this turn (possibly empty).

    The audience split is PHYSICAL, not a runtime filter: an ORCHESTRATOR turn
    registers the ``scufris`` agentic server plus the ``den`` life server (``den``
    only when a den is configured), and a regular sub-AGENT turn (``agent_id``
    set) registers ONLY the ``agent`` callback server - so a sub-agent can never
    reach the orchestrator/den tools, because those servers are simply not on its
    turn. ``is_orchestrator`` wins over ``agent_id`` (the landing orchestrator is
    never a regular agent). Returns ``[]`` when tools are disabled, or for a
    sub-agent turn with no id (nothing to address the callbacks back to).

    ``orch_session_id`` is the orchestrator's CURRENT session (the id this turn is
    resuming), injected as ``SCUFRIS_ORCH_SESSION_ID`` on the ``scufris`` server so
    ``message_agent`` / ``run_agent`` can stamp a spawned child with the chat that
    launched it and ``pending_agents`` can route escalations back to it (part 3).
    Empty on a fresh turn (no resumed id yet) - the child is then unattributed.
    """
    if not settings.agent_tools_enabled:
        return []
    api_base = f"http://{settings.host}:{settings.port}"
    command = sys.executable
    disabled = ",".join(settings.disabled_tools) if settings.disabled_tools else ""
    # The machine credential for the dashboard's own HTTP API, minted per process
    # by create_app onto ITS settings object (never os.environ - see
    # `Settings.auth_api_token`). Only the servers that CALL the API carry it
    # (`scufris` and the sub-agent `agent` callback server) - the den server does
    # not talk to the API, so it has no business holding a credential for it.
    # Empty when no app is running (a bare `scufris mcp-server` for probing),
    # which simply means the tools authenticate with nothing and are refused by a
    # gated dashboard.
    api_token = settings.auth_api_token
    servers: list[ScufrisMcpServer] = []
    if is_orchestrator:
        scufris_env: dict[str, str] = {"SCUFRIS_API_BASE": api_base}
        if api_token:
            scufris_env[API_TOKEN_ENV] = api_token
        if orch_session_id:
            scufris_env["SCUFRIS_ORCH_SESSION_ID"] = orch_session_id
        if disabled:
            scufris_env["SCUFRIS_DISABLED_TOOLS"] = disabled
        servers.append(
            ScufrisMcpServer(
                "scufris", command, ("-m", "scufris.mcp_server"), scufris_env
            )
        )
        # The den (`the-den`) server is orchestrator-only AND opt-in: registered
        # only when a den is configured, and ONLY it carries the den path, so a
        # project sub-agent can never reach the operator's journal. The operator's
        # disabled-tool set applies here too (den tools are hidable).
        if settings.den_path is not None:
            den_env = {"SCUFRIS_DEN_PATH": str(settings.den_path)}
            if disabled:
                den_env["SCUFRIS_DISABLED_TOOLS"] = disabled
            servers.append(
                ScufrisMcpServer(
                    "den", command, ("-m", "scufris.den_mcp_server"), den_env
                )
            )
    elif agent_id:
        agent_env = {"SCUFRIS_API_BASE": api_base, "SCUFRIS_AGENT_ID": agent_id}
        if api_token:
            agent_env[API_TOKEN_ENV] = api_token
        servers.append(
            ScufrisMcpServer(
                "agent",
                command,
                ("-m", "scufris.agent_mcp_server"),
                agent_env,
            )
        )
    return servers


def _mcp_overrides(
    settings: Settings,
    *,
    is_orchestrator: bool = False,
    agent_id: str = "",
    orch_session_id: str = "",
) -> list[str]:
    """`-c` config registering the MCP servers for this invocation.

    Injected on the `codex app-server` argv so nothing is written to `~/.codex`.
    The built-in scufris servers come from the shared ``scufris_mcp_servers`` core
    (an orchestrator turn gets ``scufris`` + ``den``; a sub-agent turn gets only
    ``agent``), so codex and claude never drift on which servers a turn exposes;
    codex formats each to `-c mcp_servers.<id>.*` overrides here. The audience
    split is PHYSICAL - a sub-agent simply has no ``scufris``/``den`` server - so a
    regular agent gets no other scufris tools and draws the rest from its project
    config/skills. For an unattended codex run, MCP tool calls
    would otherwise be auto-cancelled (no stdin to approve on), so trusted servers
    auto-approve their tools and approval_policy is never. The sandbox (set per
    turn on thread/start|resume) remains the real guardrail.
    """
    if not settings.agent_tools_enabled:
        return []
    args: list[str] = []
    servers = scufris_mcp_servers(
        settings,
        is_orchestrator=is_orchestrator,
        agent_id=agent_id,
        orch_session_id=orch_session_id,
    )
    for server in servers:
        args += _server_override(
            server.server_id,
            server.command,
            list(server.args),
            approve=True,
            env=server.env,
        )
    args += ["-c", 'approval_policy="never"']
    return args


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
      pointing at host_stats / disk_usage / list_processes;
    - a sub-agent that ACTUALLY holds the callbacks (the ``agent`` server:
      ``agent_id`` set) gets ``AGENT_STEERING_PREAMBLE``, telling it to signal when
      blocked;
    - any other turn - one with no audience or a tools-disabled turn - is left
      unsteered.
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
