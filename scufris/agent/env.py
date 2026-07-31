"""Finding the codex binary, building every agent child's environment, logging in.

``agent_subprocess_env`` is the single seam through which EVERY agent child
process gets its environment; nothing else may hand one to a spawn.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import Callable

from ..config import SECRET_ENV_VARS, Settings
from .events import AgentUnavailable


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
    in-process and never put in os.environ, so stripping it guards against a
    stale shell. The hostd secret is the opposite: it ARRIVES through the
    environment, because that is how a sops secret reaches the unit, so without
    this the model holds the credential for the root helper's socket and can
    apply host actions with no operator approval at all. See
    config.SECRET_ENV_VARS.

    It is a SEAM rather than a call-site strip because the call-site version was
    already forgotten once: the first fix stripped codex's environment and the
    claude backend went on spawning with no ``env=`` at all.
    ``test_no_agent_subprocess_is_spawned_without_the_stripped_environment``
    fails on any agent spawn that does not pass this, so a backend added later
    is covered by the test rather than by someone remembering.
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
