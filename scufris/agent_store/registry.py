"""``SessionRegistry``: the only home of session ids and session ownership."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from ..config import Settings

logger = logging.getLogger(__name__)


class SessionRegistry:
    """The persisted `(agent_id -> backend session history)` mapping - the ONLY
    home of session ids AND session ownership, for ALL agents (the orchestrator
    included).

    Each entry is
    ``{backend, session_id (current | None), sessions: [id,...], parent_agent_id}``.
    It records the backend the ids belong to, because a session id is
    backend-specific (a codex rollout id means nothing to claude): every
    accessor returns nothing on a backend mismatch, so a
    stale cross-backend id is structurally unreachable, and a backend switch
    starts a fresh history. ``sessions`` is the full set of sessions the agent
    has owned under ``backend`` - the switcher lists from THIS, so it never has
    to infer ownership from a provider disk scan. ``parent_agent_id`` (who
    spawned this agent) is stored and preserved but not yet used here.

    Persistence mirrors the other stores (one JSON file under the state dir,
    atomic write, tolerant load - including the legacy ``{backend, session_id}``
    shape, which loads as a one-element history). Not gated by
    ``settings_writable``: like the run-state mutators, it records
    server-internal run progress, not a user config edit."""

    def __init__(self, settings: Settings) -> None:
        self._path = Path(settings.state_dir) / "sessions.json"
        self._sessions: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("session registry: cannot read %s: %s", self._path, exc)
            return
        if not isinstance(data, dict):
            return
        for agent_id, entry in data.items():
            if not (isinstance(agent_id, str) and isinstance(entry, dict)):
                continue
            backend = entry.get("backend")
            if not isinstance(backend, str):
                continue
            session_id = entry.get("session_id")
            session_id = session_id if isinstance(session_id, str) else None
            raw_sessions = entry.get("sessions")
            if isinstance(raw_sessions, list):
                sessions = [s for s in raw_sessions if isinstance(s, str)]
            elif session_id is not None:
                sessions = [session_id]  # legacy {backend, session_id} shape
            else:
                sessions = []
            parent = entry.get("parent_agent_id")
            parent_session = entry.get("parent_session_id")
            self._sessions[agent_id] = {
                "backend": backend,
                "session_id": session_id,
                "sessions": sessions,
                "parent_agent_id": parent if isinstance(parent, str) else None,
                "parent_session_id": (
                    parent_session if isinstance(parent_session, str) else None
                ),
            }

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._sessions, indent=2, sort_keys=True))
        os.replace(tmp, self._path)

    def _entry(self, agent_id: str, backend: str) -> dict[str, Any] | None:
        """The agent's entry IF it belongs to ``backend``, else None (so a
        cross-backend id is unreachable everywhere, not just in ``get``)."""
        entry = self._sessions.get(agent_id)
        if entry is None or entry["backend"] != backend:
            return None
        return entry

    def _fresh(self, agent_id: str, backend: str, session_id: str | None) -> None:
        """Replace the agent's entry with a fresh history under ``backend``,
        preserving the spawn parent (agent + session) if one was recorded - the
        parent is a fact about who spawned the child, independent of which backend
        it later runs under, so a backend switch must not drop it."""
        prev = self._sessions.get(agent_id)
        self._sessions[agent_id] = {
            "backend": backend,
            "session_id": session_id,
            "sessions": [session_id] if session_id else [],
            "parent_agent_id": prev.get("parent_agent_id") if prev else None,
            "parent_session_id": prev.get("parent_session_id") if prev else None,
        }

    def get(self, agent_id: str, backend: str) -> str | None:
        """The agent's current session id under ``backend``, or None when there
        is no mapping or the stored ids belong to another backend."""
        entry = self._entry(agent_id, backend)
        return entry["session_id"] if entry is not None else None

    def sessions_for(self, agent_id: str, backend: str) -> list[str]:
        """The agent's full session history under ``backend`` (the switcher list),
        or ``[]`` on a backend mismatch / no mapping."""
        entry = self._entry(agent_id, backend)
        return list(entry["sessions"]) if entry is not None else []

    def has(self, agent_id: str) -> bool:
        """Whether ANY mapping exists for this agent (backend-agnostic; the
        legacy-migration guard)."""
        return agent_id in self._sessions

    def set(self, agent_id: str, backend: str, session_id: str) -> None:
        """Back-compat alias of ``add`` (append a minted session + re-current)."""
        self.add(agent_id, backend, session_id)

    def add(self, agent_id: str, backend: str, session_id: str) -> None:
        """Record a newly-minted session: set it current AND append it to the
        history (deduped). A backend change starts a fresh history."""
        entry = self._entry(agent_id, backend)
        if entry is None:
            self._fresh(agent_id, backend, session_id)
        else:
            entry["session_id"] = session_id
            if session_id not in entry["sessions"]:
                entry["sessions"].append(session_id)
        self._persist()

    def set_current(self, agent_id: str, backend: str, session_id: str | None) -> None:
        """Switch to (or, with None, clear) the current session WITHOUT dropping
        history - this is "new chat" / "switch chat". A switched-to id not yet in
        the history is appended; a backend change starts a fresh history."""
        entry = self._entry(agent_id, backend)
        if entry is None:
            self._fresh(agent_id, backend, session_id)
        else:
            entry["session_id"] = session_id
            if session_id and session_id not in entry["sessions"]:
                entry["sessions"].append(session_id)
        self._persist()

    def remove(self, agent_id: str, backend: str, session_id: str) -> None:
        """Drop one session from the agent's history (a session delete), clearing
        ``current`` if it was that id. No-op on a backend mismatch / unknown id."""
        entry = self._entry(agent_id, backend)
        if entry is None:
            return
        changed = False
        if session_id in entry["sessions"]:
            entry["sessions"].remove(session_id)
            changed = True
        if entry["session_id"] == session_id:
            entry["session_id"] = None
            changed = True
        if changed:
            self._persist()

    def clear(self, agent_id: str) -> None:
        if self._sessions.pop(agent_id, None) is not None:
            self._persist()

    def set_parent(
        self,
        agent_id: str,
        parent_agent_id: str | None,
        parent_session_id: str | None,
    ) -> None:
        """Record which agent + session spawned ``agent_id``. Parent is a
        backend-independent fact, so this works even when the child has no session
        entry yet: a minimal placeholder is created (backend "") and later upgraded
        in place by ``_fresh`` when the child actually runs (which preserves the
        parent). Persists."""
        entry = self._sessions.get(agent_id)
        if entry is None:
            entry = {
                "backend": "",
                "session_id": None,
                "sessions": [],
                "parent_agent_id": None,
                "parent_session_id": None,
            }
            self._sessions[agent_id] = entry
        entry["parent_agent_id"] = parent_agent_id
        entry["parent_session_id"] = parent_session_id
        self._persist()

    def parent_of(self, agent_id: str) -> tuple[str | None, str | None]:
        """The ``(parent_agent_id, parent_session_id)`` recorded for ``agent_id``,
        or ``(None, None)``. Backend-agnostic (parent is not session-specific)."""
        entry = self._sessions.get(agent_id)
        if entry is None:
            return (None, None)
        return (entry.get("parent_agent_id"), entry.get("parent_session_id"))
