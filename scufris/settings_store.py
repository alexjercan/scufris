"""Runtime-mutable settings, layered over the env-seeded base.

``Settings`` is loaded once from the environment (the first-boot seed). The
``SettingsStore`` layers a persisted set of OVERRIDES on top, so the operator
can change whitelisted knobs from the settings page and have them stick across
restarts without editing ``.env``. Overrides live in a JSON file under the
state dir, in a profile-ready shape (``{active, profiles: {<name>: {...}}}``)
so named profiles are an additive follow-up rather than a rewrite.

Only whitelisted, safe-to-mutate keys may be overridden - never secrets or
bind addresses (``openai_api_key``, ``codex_bin``, ``codex_home``, ``host``,
``port``). Writes are refused entirely when ``settings.settings_writable`` is
false. Each write mutates the live ``Settings`` object in place (validated by
``validate_assignment``), so per-turn readers (the agent) and the config
endpoints see the new value immediately; keys that need the agent rebuilt
(``agent_enabled``/``agent_backend``) are reported via ``on_change``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import Settings

logger = logging.getLogger(__name__)

# The only settings the operator may change at runtime. Everything else stays
# env-only. Kept as a module constant so the endpoint and tests share one list.
WRITABLE_KEYS: frozenset[str] = frozenset(
    {
        "agent_enabled",
        "agent_backend",
        "agent_model",
        "agent_tools_enabled",
        "agent_timeout_seconds",
        "poll_seconds",
        "mcp_servers",
    }
)

# Changing one of these needs the agent instance rebuilt (they are read at
# build time, not per turn); the store reports them through ``on_change``.
REBUILD_KEYS: frozenset[str] = frozenset({"agent_enabled", "agent_backend"})

DEFAULT_PROFILE = "default"


class SettingsReadOnly(RuntimeError):
    """Raised when a write is attempted while ``settings_writable`` is false."""


class UnknownSettingKey(ValueError):
    """Raised when a write targets a key outside ``WRITABLE_KEYS``."""


class SettingsStore:
    """Owns the live ``Settings`` and its persisted overrides."""

    def __init__(
        self,
        settings: Settings,
        *,
        on_change: Callable[[set[str]], None] | None = None,
    ) -> None:
        self._settings = settings
        self._on_change = on_change
        self._path = Path(settings.state_dir) / "settings.json"
        self._active = DEFAULT_PROFILE
        self._profiles: dict[str, dict[str, Any]] = {DEFAULT_PROFILE: {}}
        self._load()

    @property
    def settings(self) -> Settings:
        """The current effective settings (env base with overrides applied)."""
        return self._settings

    @property
    def writable(self) -> bool:
        return bool(self._settings.settings_writable)

    def _overrides(self) -> dict[str, Any]:
        return self._profiles.setdefault(self._active, {})

    def _load(self) -> None:
        """Read persisted overrides (if any) and apply the active profile."""
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("settings store: cannot read %s: %s", self._path, exc)
            return
        active = data.get("active")
        profiles = data.get("profiles")
        if isinstance(profiles, dict):
            self._profiles = {
                name: dict(ov) for name, ov in profiles.items() if isinstance(ov, dict)
            }
        if isinstance(active, str) and active in self._profiles:
            self._active = active
        self._profiles.setdefault(self._active, {})
        # Apply the active profile's overrides onto the live settings, skipping
        # any key that no longer validates (a stale/hand-edited file must not
        # crash the server).
        for key, value in list(self._overrides().items()):
            if key not in WRITABLE_KEYS:
                logger.warning("settings store: dropping non-writable key %r", key)
                self._overrides().pop(key, None)
                continue
            try:
                setattr(self._settings, key, value)
            except ValidationError as exc:
                logger.warning("settings store: dropping invalid %r: %s", key, exc)
                self._overrides().pop(key, None)

    def apply(self, updates: dict[str, Any]) -> Settings:
        """Validate, apply and persist ``updates``; return the live settings.

        Raises ``SettingsReadOnly`` when writes are disabled, ``UnknownSettingKey``
        for a non-whitelisted key, and ``pydantic.ValidationError`` for a bad
        value (the endpoint maps each to a status code). The mutation is
        transactional: on any failure the live settings are rolled back.
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
        # e.g. mcp_servers round-trips as plain dicts.
        dumped = self._settings.model_dump(mode="json")
        overrides = self._overrides()
        for key in updates:
            overrides[key] = dumped[key]
        self._persist()
        changed = set(updates)
        if self._on_change is not None and (changed & REBUILD_KEYS):
            self._on_change(changed)
        return self._settings

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"active": self._active, "profiles": self._profiles}
        # Write to a temp file then atomically replace, so a crash mid-write
        # cannot leave a truncated settings.json (which _load would then drop).
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, self._path)
