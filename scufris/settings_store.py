"""Runtime-mutable settings, layered over the env-seeded base.

``Settings`` is loaded once from the environment (the first-boot seed). The
``SettingsStore`` layers a persisted set of OVERRIDES on top, so the operator
can change whitelisted knobs from the settings page and have them stick across
restarts without editing ``.env``. Overrides are ``settings_override`` ROWS, one
per key, holding the setting's JSON form.

Only whitelisted, safe-to-mutate keys may be overridden - never secrets or
bind addresses (``openai_api_key``, ``codex_bin``, ``codex_home``, ``host``,
``port``). Writes are refused entirely when ``settings.settings_writable`` is
false. Each write mutates the live ``Settings`` object in place (validated by
``validate_assignment``), so per-turn readers (the agent) and the config
endpoints see the new value immediately; keys that need the agent rebuilt
(``agent_enabled``/``agent_backend``) are reported via ``on_change``.

LOAD IS TOLERANT AND THE IMPORT IS NOT, deliberately. A row whose key is no
longer writable, or whose value no longer validates, is logged and skipped at
load: the alternative is a server that refuses to boot because a knob it no
longer has was set months ago, and the operator's way out of that would be to
hand-edit the very database the boot failure is denying them access to. The
legacy importer in ``db/legacy/`` refuses the same value instead, because
there the operator still has their `settings.json` in front of them, the failure
names the key, and a repaired file imports on the next run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError
from sqlalchemy import delete, insert, select

from .config import Settings
from .db import Database
from .db.models import SettingsOverrideRow

logger = logging.getLogger(__name__)

# The only settings the operator may change at runtime. Everything else stays
# env-only. Kept as a module constant so the endpoint and tests share one list.
WRITABLE_KEYS: frozenset[str] = frozenset(
    {
        "agent_enabled",
        "agent_backend",
        "agent_model",
        "claude_model",
        "agent_permission_mode",
        "agent_tools_enabled",
        "agent_timeout_seconds",
        "poll_seconds",
        "disabled_tools",
        # The scheduled host checks: enable/disable, when, and where the lines are.
        # All runtime-editable on purpose - the digest is tuned by living with it,
        # and a threshold that needs an env edit and a restart never gets tuned.
        "host_checks_enabled",
        "host_watch_enabled",
        "host_watch_interval_seconds",
        "host_digest_enabled",
        "host_digest_at",
        "host_digest_muted_until",
        "check_disk_warn_percent",
        "check_disk_crit_percent",
        "check_temp_warn_celsius",
        "check_store_dead_paths",
        "check_flake_age_days",
        "check_escalate_gc",
    }
)

# Changing one of these needs the agent instance rebuilt (they are read at
# build time, not per turn); the store reports them through ``on_change``.
REBUILD_KEYS: frozenset[str] = frozenset({"agent_enabled", "agent_backend"})


class SettingsReadOnly(RuntimeError):
    """Raised when a write is attempted while ``settings_writable`` is false."""


class UnknownSettingKey(ValueError):
    """Raised when a write targets a key outside ``WRITABLE_KEYS``."""


class SettingsStore:
    """Owns the live ``Settings`` and its persisted overrides."""

    def __init__(
        self,
        settings: Settings,
        db: Database,
        *,
        on_change: Callable[[set[str]], None] | None = None,
    ) -> None:
        self._settings = settings
        self._db = db
        self._on_change = on_change
        self._load()

    @property
    def settings(self) -> Settings:
        """The current effective settings (env base with overrides applied)."""
        return self._settings

    @property
    def writable(self) -> bool:
        return bool(self._settings.settings_writable)

    def _load(self) -> None:
        """Read the persisted overrides and apply them onto the live settings.

        A key that is no longer writable, or a value that no longer validates, is
        logged and skipped rather than raised - see the module docstring for why
        the boot path tolerates what the importer refuses. The row is LEFT in
        place: it is the operator's record of what they set, and deleting it
        during a boot they did not ask to change anything in would lose that.
        """
        for key, value in self._stored().items():
            if key not in WRITABLE_KEYS:
                logger.warning("settings store: dropping non-writable key %r", key)
                continue
            try:
                setattr(self._settings, key, value)
            except ValidationError as exc:
                logger.warning("settings store: dropping invalid %r: %s", key, exc)

    def _stored(self) -> dict[str, Any]:
        """Every persisted override, decoded from its JSON form."""
        with self._db.transaction() as conn:
            rows = conn.execute(
                select(SettingsOverrideRow.key, SettingsOverrideRow.value)
            ).all()
        overrides: dict[str, Any] = {}
        for row in rows:
            try:
                overrides[row.key] = json.loads(row.value)
            except ValueError as exc:
                logger.warning(
                    "settings store: dropping unreadable %r: %s", row.key, exc
                )
        return overrides

    def apply(self, updates: dict[str, Any]) -> Settings:
        """Validate, apply and persist ``updates``; return the live settings.

        Raises ``SettingsReadOnly`` when writes are disabled, ``UnknownSettingKey``
        for a non-whitelisted key, and ``pydantic.ValidationError`` for a bad
        value (the endpoint maps each to a status code). The mutation is
        transactional twice over: on any validation failure the live settings are
        rolled back before anything is written, and the rows themselves are
        written in ONE transaction, so a set of knobs an operator changed
        together is never half-persisted.

        The overrides are no longer read, merged and rewritten as a whole
        document: each key is its own row, so this touches exactly the keys in
        ``updates`` and two operators changing different knobs cannot lose each
        other's write.
        """
        if not self.writable:
            raise SettingsReadOnly("settings are read-only on this server")
        if not updates:
            return self._settings
        bad = set(updates) - WRITABLE_KEYS
        if bad:
            raise UnknownSettingKey(
                f"cannot set {sorted(bad)}; writable keys are {sorted(WRITABLE_KEYS)}"
            )
        old = {key: getattr(self._settings, key) for key in updates}
        applied: list[str] = []
        try:
            for key, value in updates.items():
                setattr(self._settings, key, value)
                applied.append(key)
        except ValidationError:
            for key in applied:
                setattr(self._settings, key, old[key])
            raise
        # Persist the JSON form read back from the now-coerced settings, so
        # e.g. disabled_tools round-trips as a plain list.
        dumped = self._settings.model_dump(mode="json")
        with self._db.transaction() as conn:
            for key in updates:
                conn.execute(
                    delete(SettingsOverrideRow).where(SettingsOverrideRow.key == key)
                )
                conn.execute(
                    insert(SettingsOverrideRow).values(
                        key=key, value=json.dumps(dumped[key])
                    )
                )
        changed = set(updates)
        if self._on_change is not None and (changed & REBUILD_KEYS):
            self._on_change(changed)
        return self._settings
