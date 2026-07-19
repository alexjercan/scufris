"""Tests for the agent backend.

These fake the openai-codex SDK boundary (the injectable ``open_client`` seam),
so no SDK, codex binary, or network is needed. Real device-code login and a
live model call are the operator's to run - see the task's HONEST SCOPE.
"""

from __future__ import annotations

import pytest

from scufris.agent import (
    Agent,
    AgentReply,
    AgentUnavailable,
    CodexAgent,
    DisabledAgent,
    _default_open_client,
    build_agent,
)
from scufris.config import Settings


class FakeTurnResult:
    def __init__(self, text: str) -> None:
        self.final_response = text
        self.status = "completed"


class FakeThread:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def run(self, prompt: str) -> FakeTurnResult:
        self.prompts.append(prompt)
        return FakeTurnResult(f"echo: {prompt}")


class FakeClient:
    def __init__(self) -> None:
        self.closed = False
        self.thread = FakeThread()
        self.started = 0

    async def thread_start(self, *, model: str | None, sandbox: str) -> FakeThread:
        self.started += 1
        self.last_model = model
        self.last_sandbox = sandbox
        return self.thread

    async def close(self) -> None:
        self.closed = True


def _enabled_settings() -> Settings:
    return Settings(agent_enabled=True, agent_model="gpt-5.5")


def test_build_agent_returns_disabled_when_off() -> None:
    agent = build_agent(Settings(agent_enabled=False))
    assert isinstance(agent, DisabledAgent)
    assert isinstance(agent, Agent)


@pytest.mark.asyncio
async def test_disabled_agent_chat_raises() -> None:
    agent = build_agent(Settings(agent_enabled=False))
    with pytest.raises(AgentUnavailable):
        await agent.chat("hello")
    await agent.aclose()  # no-op, must not raise


@pytest.mark.asyncio
async def test_codex_agent_runs_turn_and_reuses_thread() -> None:
    client = FakeClient()

    async def opener(_settings: Settings) -> FakeClient:
        return client

    agent = CodexAgent(_enabled_settings(), open_client=opener)
    assert isinstance(agent, Agent)

    reply = await agent.chat("hi")
    assert isinstance(reply, AgentReply)
    assert reply.text == "echo: hi"
    assert reply.status == "completed"
    assert client.last_model == "gpt-5.5"
    assert client.last_sandbox == "read-only"

    await agent.chat("again")
    # One client, one thread reused across turns (conversation continuity).
    assert client.started == 1
    assert client.thread.prompts == ["hi", "again"]

    await agent.aclose()
    assert client.closed is True


@pytest.mark.asyncio
async def test_build_agent_enabled_returns_codex_agent() -> None:
    async def opener(_settings: Settings) -> FakeClient:
        return FakeClient()

    agent = build_agent(_enabled_settings(), open_client=opener)
    assert isinstance(agent, CodexAgent)


@pytest.mark.asyncio
async def test_empty_model_passes_none_to_sdk() -> None:
    client = FakeClient()

    async def opener(_settings: Settings) -> FakeClient:
        return client

    agent = CodexAgent(Settings(agent_enabled=True, agent_model=""), open_client=opener)
    await agent.chat("x")
    assert client.last_model is None
    await agent.aclose()


@pytest.mark.asyncio
async def test_default_open_client_without_sdk_raises_unavailable() -> None:
    # openai-codex is intentionally not a pinned dependency, so the real opener
    # must fail with a clear, actionable error rather than an ImportError.
    with pytest.raises(AgentUnavailable, match="openai-codex is not installed"):
        await _default_open_client(_enabled_settings())
