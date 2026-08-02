"""Tests for the runtime settings store: overrides, persistence, gating.

The overrides are ``settings_override`` rows now, so "did that persist" is a
query rather than a file check, and every test pairs the `database` fixture with
a `Settings` whose ``state_dir`` is the same ``tmp_path``.

The two legacy shapes this store used to read on load - the profile-shaped file
and the damaged one - moved to `tests/test_db_legacy.py` with the importer that
owns them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import insert, select

from scufris.config import Settings
from scufris.db import Database
from scufris.db.models import SettingsOverrideRow
from scufris.enums import Backend
from scufris.settings_store import (
    SettingsReadOnly,
    SettingsStore,
    UnknownSettingKey,
)


def _overrides(database: Database) -> dict[str, str]:
    with database.transaction() as conn:
        return {
            row.key: row.value
            for row in conn.execute(
                select(SettingsOverrideRow.key, SettingsOverrideRow.value)
            ).all()
        }


def test_settings_store_round_trip(tmp_path: Path, database: Database) -> None:
    # Write two overrides, then a fresh Settings()+store over the same database
    # must read them back, while a non-overridden key keeps its env-base value.
    base = Settings(state_dir=tmp_path, agent_model="gpt-5.5", poll_seconds=2.0)
    SettingsStore(base, database).apply({"agent_model": "gpt-5.6", "poll_seconds": 5.0})

    fresh = Settings(state_dir=tmp_path, agent_model="gpt-5.5", poll_seconds=2.0)
    SettingsStore(fresh, database)  # applies persisted overrides on load
    assert fresh.agent_model == "gpt-5.6"
    assert fresh.poll_seconds == 5.0
    assert fresh.agent_auth_mode == "chatgpt"  # never overridden -> base value


def test_store_persists_list_override_as_json(
    tmp_path: Path, database: Database
) -> None:
    base = Settings(state_dir=tmp_path)
    SettingsStore(base, database).apply(
        {"disabled_tools": ["host_stats", "list_processes"]}
    )
    assert _overrides(database) == {
        "disabled_tools": '["host_stats", "list_processes"]'
    }
    fresh = Settings(state_dir=tmp_path)
    SettingsStore(fresh, database)
    assert fresh.disabled_tools == ["host_stats", "list_processes"]


def test_a_write_touches_only_the_keys_it_names(
    tmp_path: Path, database: Database
) -> None:
    """Two writers changing DIFFERENT knobs cannot lose each other's value.

    The file-backed store read the whole override document, merged its own key
    in and rewrote all of it, so a rewrite carried whatever it had read before
    the other write landed. One row per key removes the merge, and with it the
    window: this asserts the earlier key is still there, not that the last
    writer won.
    """
    first = Settings(state_dir=tmp_path)
    SettingsStore(first, database).apply({"poll_seconds": 7.0})

    second = Settings(state_dir=tmp_path)
    SettingsStore(second, database).apply({"agent_model": "gpt-5.6"})

    assert set(_overrides(database)) == {"poll_seconds", "agent_model"}
    fresh = Settings(state_dir=tmp_path)
    SettingsStore(fresh, database)
    assert fresh.poll_seconds == 7.0
    assert fresh.agent_model == "gpt-5.6"


def test_store_rejects_non_whitelisted_key(tmp_path: Path, database: Database) -> None:
    store = SettingsStore(Settings(state_dir=tmp_path), database)
    with pytest.raises(UnknownSettingKey):
        store.apply({"openai_api_key": "sk-secret"})
    assert _overrides(database) == {}


def test_store_refuses_writes_when_read_only(
    tmp_path: Path, database: Database
) -> None:
    store = SettingsStore(
        Settings(state_dir=tmp_path, settings_writable=False), database
    )
    with pytest.raises(SettingsReadOnly):
        store.apply({"agent_model": "gpt-5.6"})
    assert _overrides(database) == {}


def test_store_rolls_back_and_does_not_persist_bad_value(
    tmp_path: Path, database: Database
) -> None:
    base = Settings(state_dir=tmp_path, agent_backend=Backend.MOCK)
    store = SettingsStore(base, database)
    with pytest.raises(ValidationError):
        store.apply({"agent_backend": "not-a-real-backend"})
    assert base.agent_backend == "mock"  # rolled back
    assert _overrides(database) == {}


def test_store_rollback_restores_earlier_keys_on_later_failure(
    tmp_path: Path, database: Database
) -> None:
    # A valid key applied before an invalid one in the same call must be undone.
    base = Settings(
        state_dir=tmp_path, agent_model="gpt-5.5", agent_backend=Backend.MOCK
    )
    store = SettingsStore(base, database)
    with pytest.raises(ValidationError):
        store.apply({"agent_model": "gpt-5.6", "agent_backend": "nope"})
    assert base.agent_model == "gpt-5.5"
    assert base.agent_backend == "mock"
    assert _overrides(database) == {}


def test_store_on_change_fires_only_for_rebuild_keys(
    tmp_path: Path, database: Database
) -> None:
    changed: list[set[str]] = []
    store = SettingsStore(
        Settings(state_dir=tmp_path), database, on_change=lambda c: changed.append(c)
    )
    store.apply({"agent_model": "gpt-5.6"})  # not a rebuild key
    assert changed == []
    store.apply({"agent_backend": "mock"})  # rebuild key (differs from default)
    assert changed and "agent_backend" in changed[0]


def test_load_drops_a_stale_key_instead_of_refusing_to_boot(
    tmp_path: Path, database: Database
) -> None:
    """A persisted override that no longer validates must not stop the server.

    This is the tolerance the importer deliberately does NOT have: here the
    operator's only route to the bad value is the database the failure would be
    denying them, so the key is logged and skipped and everything else applies.
    The row is left in place - it is their record of what they set.
    """
    base = Settings(state_dir=tmp_path, agent_backend=Backend.MOCK)
    SettingsStore(base, database).apply({"poll_seconds": 5.0})
    with database.transaction() as conn:
        conn.execute(
            insert(SettingsOverrideRow).values(
                key="agent_backend", value='"not-a-real-backend"'
            )
        )

    fresh = Settings(state_dir=tmp_path, agent_backend=Backend.MOCK)
    SettingsStore(fresh, database)  # must not raise
    assert fresh.agent_backend == "mock"  # the bad override was skipped
    assert fresh.poll_seconds == 5.0  # the good one still applied
    assert "agent_backend" in _overrides(database)  # and the row is kept


def test_store_writable_property_reflects_setting(
    tmp_path: Path, database: Database
) -> None:
    assert SettingsStore(Settings(state_dir=tmp_path), database).writable is True
    assert (
        SettingsStore(
            Settings(state_dir=tmp_path, settings_writable=False), database
        ).writable
        is False
    )


def test_writable_keys_match_the_api_update_model() -> None:
    # WRITABLE_KEYS (store) and AgentConfigUpdate's fields (API) are two hand-kept
    # copies of the same whitelist; assert they stay in sync so a later task that
    # adds a key to one is forced to add it to the other.
    from scufris.app import AgentConfigUpdate
    from scufris.settings_store import WRITABLE_KEYS

    assert set(AgentConfigUpdate.model_fields) == set(WRITABLE_KEYS)
