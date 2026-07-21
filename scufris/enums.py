"""Shared option enums for the stringly-typed fields across the app.

All are ``enum.StrEnum`` so each member IS its string value: pydantic validates
membership on a field typed with one, but JSON/log output and ``==`` comparisons
against the raw string are unchanged (no wire-format change). Centralised here so
config, the agent store, the supervisor and the API models share ONE definition
of each fixed option set instead of duplicating ``Literal[...]`` / bare ``str``.
"""

from __future__ import annotations

from enum import StrEnum


class AuthMode(StrEnum):
    """How the agent authenticates to its backend CLI."""

    CHATGPT = "chatgpt"  # Sign in with ChatGPT subscription (primary)
    API_KEY = "api_key"  # metered API key


class Backend(StrEnum):
    """The agent backend an agent (and the landing orchestrator) runs on. Legacy
    codex MODE ids (``app_server``/``exec``) fold to ``codex`` via
    ``config.canonical_backend`` before reaching a field typed with this."""

    CODEX = "codex"
    CLAUDE = "claude"
    MOCK = "mock"  # in-process dev/test fake (behind enable_mock_backend)


class PermissionMode(StrEnum):
    """An agent's write posture, mapped to a per-backend flag in ``backends`` (codex
    ``--sandbox`` / claude ``--permission-mode``)."""

    MANUAL = "manual"  # read-only: observe/plan
    EDIT = "edit"  # may edit project files
    AUTO = "auto"  # edit + run commands unattended (full access)


class AgentState(StrEnum):
    """An agent's merged lifecycle state (run engine + rollout), for the UI."""

    IDLE = "idle"
    RUNNING = "running"
    BLOCKED = "blocked"  # waiting on an approval
    DONE = "done"
    ERROR = "error"


class RunPhase(StrEnum):
    """The supervisor run engine's low-level phase for a single supervised run
    (merged into the richer ``AgentState`` for display)."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
