"""Tests for the runtime settings store: overrides, persistence, gating."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scufris.config import McpServerSpec, Settings
from scufris.enums import Backend
from scufris.settings_store import (
    CannotDeleteProfile,
    DuplicateProfile,
    InvalidProfileName,
    SettingsReadOnly,
    SettingsStore,
    UnknownProfile,
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
    base = Settings(state_dir=tmp_path, agent_backend=Backend.MOCK)
    store = SettingsStore(base)
    with pytest.raises(ValidationError):
        store.apply({"agent_backend": "not-a-real-backend"})
    assert base.agent_backend == "mock"  # rolled back
    assert not (tmp_path / "settings.json").exists()


def test_store_rollback_restores_earlier_keys_on_later_failure(tmp_path: Path) -> None:
    # A valid key applied before an invalid one in the same call must be undone.
    base = Settings(
        state_dir=tmp_path, agent_model="gpt-5.5", agent_backend=Backend.MOCK
    )
    store = SettingsStore(base)
    with pytest.raises(ValidationError):
        store.apply({"agent_model": "gpt-5.6", "agent_backend": "nope"})
    assert base.agent_model == "gpt-5.5"
    assert base.agent_backend == "mock"


def test_store_on_change_fires_only_for_rebuild_keys(tmp_path: Path) -> None:
    changed: list[set[str]] = []
    store = SettingsStore(
        Settings(state_dir=tmp_path), on_change=lambda c: changed.append(c)
    )
    store.apply({"agent_model": "gpt-5.6"})  # not a rebuild key
    assert changed == []
    store.apply({"agent_backend": "mock"})  # rebuild key (differs from default)
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


def test_profile_switch_changes_config(tmp_path: Path) -> None:
    base = Settings(state_dir=tmp_path, agent_model="gpt-5.5")
    store = SettingsStore(base)
    # Set an override on the default profile, then branch a new profile off it
    # and change it there.
    store.apply({"agent_model": "gpt-5.6"})
    store.create_profile("cheap")
    store.activate("cheap")
    store.apply({"agent_model": "gpt-5-mini"})
    assert base.agent_model == "gpt-5-mini"
    # Switching back restores the default profile's value.
    store.activate("default")
    assert base.agent_model == "gpt-5.6"


def test_activate_resets_keys_the_target_does_not_override(tmp_path: Path) -> None:
    # default overrides poll_seconds; an empty profile must fall back to env base,
    # not keep default's value.
    base = Settings(state_dir=tmp_path, poll_seconds=2.0)
    store = SettingsStore(base)
    store.apply({"poll_seconds": 9.0})
    store.create_profile("blank", copy_from_active=False)
    store.activate("blank")
    assert base.poll_seconds == 2.0  # back to env base, not 9.0


def test_active_profile_persists(tmp_path: Path) -> None:
    base = Settings(state_dir=tmp_path, agent_model="gpt-5.5")
    store = SettingsStore(base)
    store.create_profile("cheap", copy_from_active=False)
    store.activate("cheap")
    store.apply({"agent_model": "gpt-5-mini"})
    # A fresh store over the same dir resumes the active profile.
    fresh = Settings(state_dir=tmp_path, agent_model="gpt-5.5")
    fresh_store = SettingsStore(fresh)
    assert fresh_store.active_profile == "cheap"
    assert fresh.agent_model == "gpt-5-mini"


def test_cannot_delete_active_or_last_profile(tmp_path: Path) -> None:
    store = SettingsStore(Settings(state_dir=tmp_path))
    with pytest.raises(CannotDeleteProfile):
        store.delete_profile("default")  # active and last
    store.create_profile("other")
    store.activate("other")
    with pytest.raises(CannotDeleteProfile):
        store.delete_profile("other")  # active
    store.delete_profile("default")  # non-active -> ok
    assert store.profile_names() == ["other"]


def test_create_profile_rejects_duplicate_and_bad_name(tmp_path: Path) -> None:
    store = SettingsStore(Settings(state_dir=tmp_path))
    with pytest.raises(DuplicateProfile):
        store.create_profile("default")
    with pytest.raises(InvalidProfileName):
        store.create_profile("bad name/../x")


def test_activate_unknown_profile_raises(tmp_path: Path) -> None:
    store = SettingsStore(Settings(state_dir=tmp_path))
    with pytest.raises(UnknownProfile):
        store.activate("ghost")


def test_profile_ops_refused_when_read_only(tmp_path: Path) -> None:
    store = SettingsStore(Settings(state_dir=tmp_path, settings_writable=False))
    with pytest.raises(SettingsReadOnly):
        store.create_profile("x")
    with pytest.raises(SettingsReadOnly):
        store.activate("default")


def test_activate_fires_on_change_for_rebuild_key(tmp_path: Path) -> None:
    changed: list[set[str]] = []
    base = Settings(state_dir=tmp_path, agent_backend=Backend.CODEX)
    store = SettingsStore(base, on_change=lambda c: changed.append(c))
    store.create_profile("mockp", copy_from_active=False)
    store.activate("mockp")
    store.apply({"agent_backend": "mock"})
    changed.clear()
    store.activate("default")  # back to codex -> backend changed
    assert changed and "agent_backend" in changed[0]
