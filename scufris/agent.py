"""The Scufris agent backend.

The agent is what runs tools and chats about the host. It is fronted by the
small ``Agent`` protocol so the harness is swappable; the default implementation
drives the ``codex`` CLI (nixpkgs `codex`, "Sign in with ChatGPT" subscription)
through ``codex exec`` as a subprocess.

We use the CLI rather than the ``openai-codex`` Python SDK because the SDK bundles
a prebuilt `codex` binary that does not build in the uv2nix venv (see
docs/LESSONS.md `codex-binary-breaks-uv2nix-venv`); the nixpkgs `codex` runs fine
on NixOS and shares its auth under ``CODEX_HOME``. Using a ChatGPT subscription
programmatically is a personal-use gray area (tasks/20260719-153040/SPIKE.md), so
the agent is off unless the operator enables it and has run ``codex login``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Awaitable, Callable, NamedTuple, Protocol, runtime_checkable

from pydantic import BaseModel

from .config import Settings


class AgentUnavailable(RuntimeError):
    """Raised when the agent cannot serve a request (disabled or unconfigured)."""


class AgentReply(BaseModel):
    text: str
    status: str | None = None


@runtime_checkable
class Agent(Protocol):
    """What the chat layer depends on. Implementations are swappable."""

    async def chat(self, prompt: str) -> AgentReply:
        """Run one turn and return the assistant's reply."""
        ...

    def reset(self) -> None:
        """Start a fresh conversation (forget prior context)."""
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

    def reset(self) -> None:
        return None

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


# The runner is the injectable seam: production shells out to `codex exec`; tests
# pass a fake so no codex binary or network is needed. It receives the current
# codex thread id (None for a new conversation) and returns the reply plus the
# thread id to continue from.
CodexRunner = Callable[[Settings, str, "str | None"], Awaitable[TurnOutcome]]


def _parse_thread_id(stdout: bytes) -> str | None:
    """Recover the codex thread id from the `--json` event stream (turn 1)."""
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "thread.started":
            tid = event.get("thread_id")
            if isinstance(tid, str):
                return tid
    return None


def _mcp_overrides(settings: Settings) -> list[str]:
    """`-c` config that registers the Scufris MCP server for this invocation.

    Injected per codex-exec call so nothing is written to `~/.codex`. The server
    runs with this interpreter (`python -m scufris.mcp_server`). For unattended
    `codex exec`, MCP tool calls would otherwise be auto-cancelled (no stdin to
    approve on), so we auto-approve this server's tools and set approval_policy
    to never. The read-only sandbox (set on turn 1) remains the real guardrail.
    """
    if not settings.agent_tools_enabled:
        return []
    command = json.dumps(sys.executable)
    server_args = json.dumps(["-m", "scufris.mcp_server"])
    return [
        "-c",
        f"mcp_servers.scufris.command={command}",
        "-c",
        f"mcp_servers.scufris.args={server_args}",
        "-c",
        'mcp_servers.scufris.default_tools_approval_mode="approve"',
        "-c",
        'approval_policy="never"',
    ]


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
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / "reply.txt"
        args = [codex_bin, "exec"]
        if thread_id:
            args.append("resume")
        args += ["--json", "--output-last-message", str(out_file)]
        # `resume` inherits the original session's sandbox and rejects the flag;
        # only set it on the first turn (it persists to resumes).
        if not thread_id:
            args += ["--sandbox", "read-only"]
        args += ["--skip-git-repo-check"]
        args += _mcp_overrides(settings)
        if settings.agent_model:
            args += ["--model", settings.agent_model]
        if thread_id:
            args.append(thread_id)
        args.append(prompt)

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
            raise AgentUnavailable(
                f"codex exec timed out after {settings.agent_timeout_seconds}s"
            ) from exc

        if proc.returncode != 0:
            detail = (
                stderr.decode(errors="replace").strip() or f"exit {proc.returncode}"
            )
            raise AgentUnavailable(f"codex exec failed: {detail}")

        new_thread_id = _parse_thread_id(stdout) or thread_id
        try:
            text = out_file.read_text().strip()
        except OSError:
            text = ""
        return TurnOutcome(text=text, thread_id=new_thread_id)


class CodexCliAgent:
    """Drive Codex via `codex exec`, keeping one conversation across turns.

    The codex thread id is remembered so turns share context; ``reset`` starts a
    fresh conversation.
    """

    def __init__(
        self, settings: Settings, runner: CodexRunner = _run_codex_exec
    ) -> None:
        self._settings = settings
        self._runner = runner
        self._thread_id: str | None = None

    async def chat(self, prompt: str) -> AgentReply:
        outcome = await self._runner(self._settings, prompt, self._thread_id)
        self._thread_id = outcome.thread_id
        return AgentReply(text=outcome.text, status="completed")

    def reset(self) -> None:
        self._thread_id = None

    async def aclose(self) -> None:
        return None


def build_agent(settings: Settings, runner: CodexRunner = _run_codex_exec) -> Agent:
    """Select the agent implementation from settings."""
    if not settings.agent_enabled:
        return DisabledAgent(
            "agent is disabled. Set SCUFRIS_AGENT_ENABLED=1 and run `codex login` "
            "(or `scufris login`) to enable it."
        )
    return CodexCliAgent(settings, runner=runner)


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
