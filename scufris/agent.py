"""The Scufris agent backend.

The agent is what runs tools and chats about the host. It is fronted by the
small ``Agent`` protocol so the harness/provider is swappable; the default
implementation drives OpenAI Codex through the ``openai-codex`` Python SDK, with
"Sign in with ChatGPT" (subscription) as the primary auth and an API key as the
fallback.

The SDK is imported lazily and is NOT a pinned project dependency: it bundles a
prebuilt `codex` CLI binary that does not build cleanly in the uv2nix venv (see
tasks/20260719-153040/SPIKE.md and the NixOS-runtime follow-up), and using a
ChatGPT subscription programmatically is a personal-use gray area. So the app
runs with the agent DISABLED unless the operator installs the toolchain, sets
``SCUFRIS_AGENT_ENABLED=1`` and runs ``scufris login``.

Live auth and a real model call are the operator's to perform; everything here
is exercised in tests against a faked SDK boundary.
"""

from __future__ import annotations

import os
from typing import Awaitable, Callable, Protocol, runtime_checkable

from pydantic import BaseModel

from .config import Settings

# Codex sandbox value: the chat agent is read-only, it does not edit the host.
_SANDBOX_READ_ONLY = "read-only"


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
        """Release any underlying resources (the Codex client/subprocess)."""
        ...


class DisabledAgent:
    """Stand-in when the agent is off or unconfigured.

    Every call fails with an actionable message rather than pretending to work.
    """

    def __init__(self, reason: str) -> None:
        self._reason = reason

    async def chat(self, prompt: str) -> AgentReply:
        raise AgentUnavailable(self._reason)

    async def aclose(self) -> None:
        return None


# A client opener is the injectable seam: production imports openai-codex and
# authenticates; tests pass a fake so no SDK/binary is needed.
ClientOpener = Callable[[Settings], Awaitable["object"]]


async def _default_open_client(settings: Settings) -> object:
    """Create and authenticate a real ``openai_codex.AsyncCodex`` client."""
    if settings.codex_home is not None:
        os.environ["CODEX_HOME"] = str(settings.codex_home)
    try:
        import openai_codex
    except ImportError as exc:  # pragma: no cover - exercised via the seam
        raise AgentUnavailable(
            "openai-codex is not installed. Install the Codex toolchain "
            "(see README) to enable the agent."
        ) from exc

    client = openai_codex.AsyncCodex()
    if settings.agent_auth_mode == "api_key":
        if not settings.openai_api_key:
            await client.close()
            raise AgentUnavailable(
                "agent_auth_mode=api_key but SCUFRIS_OPENAI_API_KEY is unset."
            )
        await client.login_api_key(settings.openai_api_key)
    # chatgpt mode relies on a prior `scufris login` device-code auth that Codex
    # persisted under CODEX_HOME; nothing to do here beyond constructing.
    return client


class CodexAgent:
    """Drive OpenAI Codex via the openai-codex SDK.

    A single Codex thread is started lazily and reused across turns so the
    conversation keeps context; the client/subprocess is torn down by
    ``aclose()``.
    """

    def __init__(
        self, settings: Settings, open_client: ClientOpener = _default_open_client
    ) -> None:
        self._settings = settings
        self._open_client = open_client
        self._client: object | None = None
        self._thread: object | None = None

    async def _ensure_thread(self) -> object:
        if self._client is None:
            self._client = await self._open_client(self._settings)
        if self._thread is None:
            self._thread = await self._client.thread_start(  # type: ignore[attr-defined]
                model=self._settings.agent_model or None,
                sandbox=_SANDBOX_READ_ONLY,
            )
        return self._thread

    async def chat(self, prompt: str) -> AgentReply:
        thread = await self._ensure_thread()
        result = await thread.run(prompt)  # type: ignore[attr-defined]
        text = getattr(result, "final_response", None) or ""
        return AgentReply(text=text, status=getattr(result, "status", None))

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()  # type: ignore[attr-defined]
        self._client = None
        self._thread = None


def build_agent(
    settings: Settings, open_client: ClientOpener = _default_open_client
) -> Agent:
    """Select the agent implementation from settings."""
    if not settings.agent_enabled:
        return DisabledAgent(
            "agent is disabled. Set SCUFRIS_AGENT_ENABLED=1 and run "
            "`scufris login` to enable it."
        )
    return CodexAgent(settings, open_client=open_client)


async def login(settings: Settings, *, printer: Callable[[str], None] = print) -> None:
    """Authenticate Codex for this host (operator step, run via ``scufris login``).

    In chatgpt mode this runs the device-code flow: it prints a URL and code the
    operator enters in a browser, then blocks until login completes. In api_key
    mode it stores the configured key.
    """
    if settings.codex_home is not None:
        os.environ["CODEX_HOME"] = str(settings.codex_home)
    try:
        import openai_codex
    except ImportError as exc:  # pragma: no cover - exercised via docs, not CI
        raise AgentUnavailable(
            "openai-codex is not installed. Install the Codex toolchain "
            "(see README) to enable the agent."
        ) from exc

    client = openai_codex.AsyncCodex()
    try:
        if settings.agent_auth_mode == "api_key":
            if not settings.openai_api_key:
                raise AgentUnavailable(
                    "agent_auth_mode=api_key but SCUFRIS_OPENAI_API_KEY is unset."
                )
            await client.login_api_key(settings.openai_api_key)
            printer("Logged in with API key.")
        else:
            handle = await client.login_chatgpt_device_code()
            printer(
                f"Open {handle.verification_url} and enter code: {handle.user_code}"
            )
            printer("Waiting for you to complete sign-in...")
            await handle.wait()
            printer("Sign in with ChatGPT complete.")
    finally:
        await client.close()
