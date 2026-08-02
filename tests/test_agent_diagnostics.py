"""The backend-aware diagnostics surface: `/api/agents/{id}/*` for every backend.

These drive the cross-backend contract, not one backend's readers: what an agent
can report is declared by its backend adapter, so "this backend has no such
reader" (`supported: false`) is a different answer from "the reader ran and found
nothing" (`supported: true`, empty value).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from scufris.app import create_app
from scufris.config import Settings
from scufris.enums import Backend
from scufris.metrics import Collector

_BACKENDS = ("codex", "claude", "opencode", "mock")

#: Backends whose adapter reads codex-account usage/memory (the rollout readers).
_USAGE_BACKENDS = ("codex",)

#: Backends that wire the scufris MCP servers into a turn, so a tools listing is
#: something they can answer at all.
_MCP_BACKENDS = ("codex", "claude")


def _write_rollout(home: Path, session_id: str, *, used_percent: float = 42.0) -> None:
    """One fake codex rollout carrying account rate limits, so the codex usage and
    memory readers have real data to find."""
    day = home / "sessions" / "2026" / "07" / "19"
    day.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = [
        {
            "type": "session_meta",
            "payload": {
                "session_id": session_id,
                "id": session_id,
                "timestamp": "2026-07-19T14:39:30.556Z",
                "cwd": os.getcwd(),
                "originator": "codex_exec",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "model_context_window": 258400,
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 40,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                        "total_tokens": 120,
                    },
                },
                "rate_limits": {
                    "plan_type": "plus",
                    "primary": {
                        "used_percent": used_percent,
                        "window_minutes": 10080,
                        "resets_at": 1_760_000_000,
                    },
                },
            },
        },
    ]
    path = day / f"rollout-2026-07-19T14-39-30-{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _settings(tmp_path: Path, codex_home: Path, **over: Any) -> Settings:
    return Settings(
        web_dist=tmp_path / "absent",
        state_dir=tmp_path / "state",
        codex_home=codex_home,
        agent_enabled=True,
        enable_mock_backend=True,  # so a mock agent can be created
        **over,
    )


def _client_with_agents(
    collector: Collector, settings: Settings, tmp_path: Path
) -> tuple[TestClient, dict[str, str]]:
    """A client plus one agent per backend, all in one project."""
    client = TestClient(create_app(collector=collector, settings=settings))
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    client.post("/api/projects", json={"name": "My App", "cwd": str(proj)})
    ids = {
        backend: client.post(
            "/api/agents",
            json={"name": backend, "project_id": "my-app", "backend": backend},
        ).json()["id"]
        for backend in _BACKENDS
    }
    return client, ids


def test_scoped_diagnostics_are_backend_consistent(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """Every backend answers the SAME scoped surface with the same shapes: an
    account carrying the agent's own backend/model/auth mode, and a capability
    envelope for usage, memory and the visible-tools listing. The orchestrator is
    one more agent on that surface, not a special case."""
    home = tmp_path / "codex"
    _write_rollout(home, "sess-consistent")
    client, ids = _client_with_agents(
        fake_collector, _settings(tmp_path, home), tmp_path
    )

    for backend in _BACKENDS:
        agent_id = ids[backend]
        account = client.get(f"/api/agents/{agent_id}/account").json()
        assert set(account) == {"auth_mode", "model", "enabled", "quota"}
        assert account["model"]
        assert set(account["quota"]) == {"supported", "value"}

        for panel in ("usage", "memory", "tools"):
            body = client.get(f"/api/agents/{agent_id}/{panel}").json()
            assert set(body) == {"supported", "value"}, (backend, panel)
            assert isinstance(body["supported"], bool)
            if not body["supported"]:
                assert body["value"] is None, (backend, panel)

        health = client.get(f"/api/agents/{agent_id}/health").json()
        assert health["backend"] == backend

        # The capability answers agree with the backend's declared readers.
        usage = client.get(f"/api/agents/{agent_id}/usage").json()
        memory = client.get(f"/api/agents/{agent_id}/memory").json()
        tools = client.get(f"/api/agents/{agent_id}/tools").json()
        assert usage["supported"] is (backend in _USAGE_BACKENDS), backend
        assert memory["supported"] is (backend in _USAGE_BACKENDS), backend
        assert tools["supported"] is (backend in _MCP_BACKENDS), backend

    # The orchestrator (codex by default) resolves the same shapes.
    orch_account = client.get("/api/agents/orchestrator/account").json()
    assert set(orch_account["quota"]) == {"supported", "value"}
    assert client.get("/api/agents/orchestrator/usage").json()["supported"] is True
    assert client.get("/api/agents/orchestrator/tools").json()["supported"] is True

    # Unknown agent ids still 404 on every panel: the service raises nothing
    # HTTP-shaped, the route owns the 404.
    for panel in ("usage", "memory", "account", "tools", "health"):
        assert client.get(f"/api/agents/ghost/{panel}").status_code == 404


def test_non_codex_orchestrator_hides_codex_account_data(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """A codex home full of rollouts does not leak into a claude orchestrator: its
    usage and memory report unsupported, and its account carries no quota - even
    though the same box's codex reader would find data."""
    home = tmp_path / "codex"
    _write_rollout(home, "sess-orch", used_percent=37.0)
    settings = _settings(tmp_path, home, agent_backend=Backend.CODEX)
    client, _ids = _client_with_agents(fake_collector, settings, tmp_path)

    # On codex, the data IS there - so its absence below is the backend, not the box.
    usage = client.get("/api/agents/orchestrator/usage").json()
    assert usage["value"]["primary"]["used_percent"] == 37.0
    assert (
        client.get("/api/agents/orchestrator/memory").json()["value"]["session_count"]
        >= 1
    )

    assert (
        client.patch("/api/agents/orchestrator", json={"backend": "claude"}).status_code
        == 200
    )

    usage = client.get("/api/agents/orchestrator/usage").json()
    assert usage == {"supported": False, "value": None}
    memory = client.get("/api/agents/orchestrator/memory").json()
    assert memory == {"supported": False, "value": None}
    account = client.get("/api/agents/orchestrator/account").json()
    assert account["quota"] == {"supported": False, "value": None}
    assert account["auth_mode"] == "claude_ai"


def test_unsupported_diagnostics_are_explicit(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """ "No reader" and "the reader found nothing" are different answers. A codex
    agent on an EMPTY codex home reports supported-but-empty; a mock agent reports
    unsupported. The same split applies to the visible-tools listing."""
    empty_home = tmp_path / "no-codex"
    client, ids = _client_with_agents(
        fake_collector, _settings(tmp_path, empty_home), tmp_path
    )

    # Supported, but there is nothing on disk to report.
    codex_usage = client.get(f"/api/agents/{ids['codex']}/usage").json()
    assert codex_usage == {"supported": True, "value": None}
    codex_memory = client.get(f"/api/agents/{ids['codex']}/memory").json()
    assert codex_memory["supported"] is True
    assert codex_memory["value"]["session_count"] == 0

    # Not supported at all - distinct bytes from the empty result above.
    mock_usage = client.get(f"/api/agents/{ids['mock']}/usage").json()
    assert mock_usage == {"supported": False, "value": None}
    mock_memory = client.get(f"/api/agents/{ids['mock']}/memory").json()
    assert mock_memory == {"supported": False, "value": None}
    assert mock_memory != codex_memory

    # Tools: a codex sub-agent supports the listing (and gets its callbacks); an
    # opencode agent wires no scufris MCP, so it has no listing to give.
    codex_tools = client.get(f"/api/agents/{ids['codex']}/tools").json()
    assert codex_tools["supported"] is True
    assert {t["name"] for t in codex_tools["value"]} == {"request_input", "report_back"}
    assert client.get(f"/api/agents/{ids['opencode']}/tools").json() == {
        "supported": False,
        "value": None,
    }


def test_diagnostics_follow_the_persisted_orchestrator_record(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The whole capability set moves with a backend switch, because it is read off
    the persisted orchestrator record rather than static settings: switching to
    opencode drops usage, memory AND the tools listing at once, and switching back
    restores them - with the model and auth mode following too."""
    home = tmp_path / "codex"
    _write_rollout(home, "sess-switch")
    settings = _settings(tmp_path, home, agent_backend=Backend.CODEX)
    client, _ids = _client_with_agents(fake_collector, settings, tmp_path)

    def snapshot() -> dict[str, Any]:
        account = client.get("/api/agents/orchestrator/account").json()
        return {
            "auth_mode": account["auth_mode"],
            "model": account["model"],
            "quota": account["quota"]["supported"],
            "usage": client.get("/api/agents/orchestrator/usage").json()["supported"],
            "memory": client.get("/api/agents/orchestrator/memory").json()["supported"],
            "tools": client.get("/api/agents/orchestrator/tools").json()["supported"],
        }

    before = snapshot()
    assert before["auth_mode"] == "chatgpt"
    assert (before["quota"], before["usage"], before["memory"], before["tools"]) == (
        True,
        True,
        True,
        True,
    )

    # Backend only: the model follows the record's effective backend by itself.
    assert (
        client.patch(
            "/api/agents/orchestrator", json={"backend": "opencode"}
        ).status_code
        == 200
    )
    switched = snapshot()
    assert switched["model"] == settings.opencode_model
    assert switched["auth_mode"] == "local"
    assert (
        switched["quota"],
        switched["usage"],
        switched["memory"],
        switched["tools"],
    ) == (False, False, False, False)
    # The health probe follows the record too.
    assert client.get("/api/agents/orchestrator/health").json()["backend"] == "opencode"

    client.patch("/api/agents/orchestrator", json={"backend": "codex"})
    assert snapshot() == before
