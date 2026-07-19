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
import os
import shutil
import tempfile
from pathlib import Path
from typing import Awaitable, Callable, Protocol, runtime_checkable

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

    async def aclose(self) -> None:
        """Release any resources held by the agent."""
        ...


class DisabledAgent:
    """Stand-in when the agent is off or unconfigured; every call fails clearly."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    async def chat(self, prompt: str) -> AgentReply:
        raise AgentUnavailable(self._reason)

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


# The runner is the injectable seam: production shells out to `codex exec`;
# tests pass a fake so no codex binary or network is needed.
CodexRunner = Callable[[Settings, str], Awaitable[str]]


async def _run_codex_exec(settings: Settings, prompt: str) -> str:
    """Run one non-interactive `codex exec` turn and return the final message.

    Read-only sandbox (the chat agent does not modify the host), no session
    persisted, final message captured via ``-o``.
    """
    codex_bin = _resolve_codex_bin(settings)
    with tempfile.TemporaryDirectory() as tmp:
        out_file = Path(tmp) / "reply.txt"
        args = [
            codex_bin,
            "exec",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "--output-last-message",
            str(out_file),
        ]
        if settings.agent_model:
            args += ["--model", settings.agent_model]
        args.append(prompt)

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_codex_env(settings),
        )
        try:
            _, stderr = await asyncio.wait_for(
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

        try:
            return out_file.read_text().strip()
        except OSError:
            return ""


class CodexCliAgent:
    """Drive Codex by shelling out to `codex exec` (one turn per ``chat``)."""

    def __init__(
        self, settings: Settings, runner: CodexRunner = _run_codex_exec
    ) -> None:
        self._settings = settings
        self._runner = runner

    async def chat(self, prompt: str) -> AgentReply:
        text = await self._runner(self._settings, prompt)
        return AgentReply(text=text, status="completed")

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
