"""Tests for the runtime settings store: overrides, persistence, gating."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scufris.agent import AgentHandle, MockAgent
from scufris.config import McpServerSpec, Settings
from scufris.settings_store import (
    SettingsReadOnly,
    SettingsStore,
    UnknownSettingKey,
)


def test_settings_store_round_trip(tmp_path: Path) -> None:
    # Write two overrides, then a fresh Settings()+store over the same state dir
    # must read them back, while a non-overridden key keeps its env-base value.
    base = Settings(state_dir=tmp_path, agent_model="gpt-5.5", poll_seconds=2.0)
    SettingsStore(base).apply({"agent_model": "gpt-5.6", "poll_seconds": 5.0})

    fresh = Settings(state_dir=tmp_path, agent_model="gpt-5.5", poll_seconds=2.0)
    SettingsStore(fresh)  # applies persisted overrides on load
    assert fresh.agent_model == "gpt-5.6"
    assert fresh.poll_seconds == 5.0
    assert fresh.agent_auth_mode == "chatgpt"  # never overridden -> base value


def test_store_persists_mcp_servers_as_json(tmp_path: Path) -> None:
    base = Settings(state_dir=tmp_path)
    SettingsStore(base).apply(
        {"mcp_servers": [McpServerSpec(id="extra", command="mcp-extra")]}
    )
    fresh = Settings(state_dir=tmp_path)
    SettingsStore(fresh)
    assert [s.id for s in fresh.mcp_servers] == ["extra"]
    assert fresh.mcp_servers[0].command == "mcp-extra"


def test_store_rejects_non_whitelisted_key(tmp_path: Path) -> None:
    store = SettingsStore(Settings(state_dir=tmp_path))
    with pytest.raises(UnknownSettingKey):
        store.apply({"openai_api_key": "sk-secret"})
    assert not (tmp_path / "settings.json").exists()


def test_store_refuses_writes_when_read_only(tmp_path: Path) -> None:
    store = SettingsStore(Settings(state_dir=tmp_path, settings_writable=False))
    with pytest.raises(SettingsReadOnly):
        store.apply({"agent_model": "gpt-5.6"})
    assert not (tmp_path / "settings.json").exists()


def test_store_rolls_back_and_does_not_persist_bad_value(tmp_path: Path) -> None:
    base = Settings(state_dir=tmp_path, agent_backend="exec")
    store = SettingsStore(base)
    with pytest.raises(ValidationError):
        store.apply({"agent_backend": "not-a-real-backend"})
    assert base.agent_backend == "exec"  # rolled back
    assert not (tmp_path / "settings.json").exists()


def test_store_rollback_restores_earlier_keys_on_later_failure(tmp_path: Path) -> None:
    # A valid key applied before an invalid one in the same call must be undone.
    base = Settings(state_dir=tmp_path, agent_model="gpt-5.5", agent_backend="exec")
    store = SettingsStore(base)
    with pytest.raises(ValidationError):
        store.apply({"agent_model": "gpt-5.6", "agent_backend": "nope"})
    assert base.agent_model == "gpt-5.5"
    assert base.agent_backend == "exec"


def test_store_on_change_fires_only_for_rebuild_keys(tmp_path: Path) -> None:
    changed: list[set[str]] = []
    store = SettingsStore(
        Settings(state_dir=tmp_path), on_change=lambda c: changed.append(c)
    )
    store.apply({"agent_model": "gpt-5.6"})  # not a rebuild key
    assert changed == []
    store.apply({"agent_backend": "exec"})  # rebuild key
    assert changed and "agent_backend" in changed[0]


def test_store_ignores_corrupt_state_file(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text("{not valid json")
    base = Settings(state_dir=tmp_path, agent_model="gpt-5.5")
    SettingsStore(base)  # must not raise
    assert base.agent_model == "gpt-5.5"


def test_store_writable_property_reflects_setting(tmp_path: Path) -> None:
    assert SettingsStore(Settings(state_dir=tmp_path)).writable is True
    assert (
        SettingsStore(Settings(state_dir=tmp_path, settings_writable=False)).writable
        is False
    )


def test_writable_keys_match_the_api_update_model() -> None:
    # WRITABLE_KEYS (store) and AgentConfigUpdate's fields (API) are two hand-kept
    # copies of the same whitelist; assert they stay in sync so a later task that
    # adds a key to one is forced to add it to the other.
    from scufris.app import AgentConfigUpdate
    from scufris.settings_store import WRITABLE_KEYS

    assert set(AgentConfigUpdate.model_fields) == set(WRITABLE_KEYS)


def test_agent_handle_rebuilds_and_carries_session() -> None:
    built: list[MockAgent] = []

    def builder(_settings: Settings) -> MockAgent:
        agent = MockAgent()
        built.append(agent)
        return agent

    handle = AgentHandle(Settings(agent_backend="mock"), builder)
    handle.switch_session("sess-1")
    handle.rebuild()
    assert len(built) == 2  # rebuilt a new inner agent
    assert handle.current_session_id() == "sess-1"  # session carried across
