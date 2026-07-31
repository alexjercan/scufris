"""Rollout files: where they live, how they are found, read, and summarised.

Codex records every session as a JSONL rollout under ``$CODEX_HOME/sessions``.
This module owns locating those files and iterating their events, plus the
queries that need nothing but the event stream: the session list, a session's
context snapshot, its last-write time, and its deletion. Everything here is
read-only apart from ``delete_session``, and every entry point takes an explicit
``codex_home`` so tests can drive it against a temp directory of fake rollouts
with no codex binary in sight.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..config import Settings
from .models import SessionContext, SessionInfo
from .steering import strip_steering

logger = logging.getLogger(__name__)

_EPOCH = datetime.fromtimestamp(0, timezone.utc)
_TITLE_MAX = 80

# Originators scufris tags its own sessions with. `codex exec` writes the codex
# default "codex_exec"; `codex app-server` writes the `clientInfo.name` we send
# on initialize ("scufris"). Both are ours, so the switch list must accept
# either - otherwise app-server sessions (the current default backend) vanish
# from the list even though they are on disk.
_SCUFRIS_ORIGINATORS = frozenset({"codex_exec", "scufris"})


def resolve_codex_home(settings: Settings) -> Path:
    """Where codex keeps its state: explicit setting, ``$CODEX_HOME``, or ~/.codex."""
    if settings.codex_home is not None:
        return settings.codex_home
    env = os.environ.get("CODEX_HOME")
    if env:
        return Path(env)
    return Path.home() / ".codex"


def _sessions_dir(codex_home: Path) -> Path:
    return codex_home / "sessions"


def _iter_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a rollout, skipping malformed/oversized lines."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _event_kind(event: dict[str, Any]) -> str | None:
    """The discriminating kind: ``event_msg`` -> payload.type, else top-level type."""
    etype = event.get("type")
    if etype == "event_msg":
        ptype = _payload(event).get("type")
        return ptype if isinstance(ptype, str) else None
    return etype if isinstance(etype, str) else None


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except OSError:
        return None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _read_head(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(session_meta payload, first user-message text)`` from a rollout.

    Stops as soon as both are found so a long session is not read end to end.
    """
    meta: dict[str, Any] | None = None
    title: str | None = None
    for event in _iter_events(path):
        kind = _event_kind(event)
        if kind == "session_meta" and meta is None:
            meta = _payload(event)
        elif kind == "user_message" and title is None:
            message = _payload(event).get("message")
            if isinstance(message, str):
                # Strip the agent's steering preamble so the title is the user's
                # actual first question, not the injected tool instructions.
                title = strip_steering(message).strip()
        if meta is not None and title is not None:
            break
    return meta, title


def list_sessions(codex_home: Path, cwd: str) -> list[SessionInfo]:
    """List this app's codex sessions, newest first.

    Scoped to scufris-originated sessions (see ``_SCUFRIS_ORIGINATORS``) whose
    ``cwd`` matches the server's, which mirrors codex's own default resume filter
    and keeps unrelated codex sessions (other directories, the interactive TUI)
    out of the list.
    """
    root = _sessions_dir(codex_home)
    if not root.is_dir():
        return []
    sessions: list[SessionInfo] = []
    for path in root.rglob("rollout-*.jsonl"):
        meta, title = _read_head(path)
        if meta is None:
            continue
        if meta.get("originator") not in _SCUFRIS_ORIGINATORS:
            continue
        session_cwd = meta.get("cwd")
        if not isinstance(session_cwd, str) or session_cwd != cwd:
            continue
        sid = meta.get("session_id") or meta.get("id")
        if not isinstance(sid, str):
            continue
        git = meta.get("git")
        branch = git.get("branch") if isinstance(git, dict) else None
        sessions.append(
            SessionInfo(
                id=sid,
                title=(title or "(untitled)")[:_TITLE_MAX],
                started_at=_parse_ts(meta.get("timestamp")),
                updated_at=_mtime(path),
                git_branch=branch if isinstance(branch, str) else None,
                cwd=session_cwd,
            )
        )
    sessions.sort(key=lambda s: s.updated_at or s.started_at or _EPOCH, reverse=True)
    logger.debug("list_sessions cwd=%s -> %d", cwd, len(sessions))
    return sessions


def _find_rollout(codex_home: Path, session_id: str) -> Path | None:
    """Locate a session's rollout file (its id is embedded in the filename)."""
    root = _sessions_dir(codex_home)
    if not root.is_dir():
        return None
    # session_id arrives from the client (switch), so escape glob metacharacters
    # before interpolating it into an rglob pattern - it must match literally.
    matches = list(root.rglob(f"rollout-*-{glob.escape(session_id)}.jsonl"))
    if matches:
        return matches[0]
    for path in root.rglob("rollout-*.jsonl"):
        meta, _ = _read_head(path)
        if meta and session_id in (meta.get("session_id"), meta.get("id")):
            return path
    return None


def rollout_mtime(codex_home: Path, session_id: str | None) -> float | None:
    """The last-write time of a session's rollout, or None if it cannot be read.

    Used as a read-only "last activity" signal for an agent's status - the rollout
    is appended to as the turn progresses.
    """
    if not session_id:
        return None
    path = _find_rollout(codex_home, session_id)
    if path is None:
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def delete_session(codex_home: Path, session_id: str | None) -> bool:
    """Delete a session by unlinking its rollout file. Returns True if removed.

    Only ever touches the one validated rollout inside ``CODEX_HOME`` (located via
    the glob-escaped ``_find_rollout``); a no-op for an empty/unknown id.
    """
    if not session_id:
        return False
    path = _find_rollout(codex_home, session_id)
    if path is None:
        logger.debug("delete_session %s -> not found", session_id)
        return False
    try:
        path.unlink()
    except OSError:
        logger.warning("delete_session %s -> unlink failed", session_id)
        return False
    logger.info("deleted session %s", session_id)
    return True


def read_context(codex_home: Path, session_id: str | None) -> SessionContext | None:
    """The current session's context snapshot, or ``None`` if it cannot be read."""
    if not session_id:
        return None
    path = _find_rollout(codex_home, session_id)
    if path is None:
        return None
    window = 0
    total: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    turns = 0
    tools = 0
    for event in _iter_events(path):
        kind = _event_kind(event)
        if kind == "user_message":
            turns += 1
        elif kind == "mcp_tool_call_end":
            tools += 1
        elif kind == "token_count":
            info = _payload(event).get("info")
            if isinstance(info, dict):
                candidate = info.get("model_context_window")
                if isinstance(candidate, int):
                    window = candidate
                total_usage = info.get("total_token_usage")
                if isinstance(total_usage, dict):
                    total = total_usage
                last_usage = info.get("last_token_usage")
                if isinstance(last_usage, dict):
                    last = last_usage
    context = SessionContext(
        session_id=session_id,
        context_window=window,
        turn_count=turns,
        tool_call_count=tools,
    )
    # input/cached describe the CURRENT context occupancy, which is the LAST
    # request's input (not the cumulative sum across turns, which overcounts and
    # can exceed the window). Fall back to the total if a session predates the
    # last-usage field. output/reasoning/total stay cumulative (work done).
    fill = last or total
    if fill is not None:
        context.input_tokens = _int(fill.get("input_tokens"))
        context.cached_input_tokens = _int(fill.get("cached_input_tokens"))
    if total is not None:
        context.output_tokens = _int(total.get("output_tokens"))
        context.reasoning_output_tokens = _int(total.get("reasoning_output_tokens"))
        context.total_tokens = _int(total.get("total_tokens"))
    return context
