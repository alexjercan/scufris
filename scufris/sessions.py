"""Codex session introspection.

Reads codex's on-disk rollout files (JSONL under ``$CODEX_HOME/sessions``) to
expose what the ``codex exec --json`` stream does not: the list of past sessions
(so the UI can switch between them), a per-session context snapshot (window +
token usage + turn/tool counts), and the account usage/quota (the weekly rate
limit). Everything here is read-only.

Codex already records all of this on disk (see tasks/20260719-212152/SPIKE.md),
so this harvests what exists rather than adding subprocess calls. The functions
take an explicit ``codex_home`` and ``cwd`` so tests can drive them against a
temp directory of fake rollout files, with no codex binary in sight.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel

from .config import Settings

_EPOCH = datetime.fromtimestamp(0, timezone.utc)
_TITLE_MAX = 80


class RateWindow(BaseModel):
    """One rate-limit window (codex reports a weekly primary at 10080 minutes)."""

    used_percent: float
    window_minutes: int
    resets_at: int | None = None


class UsageQuota(BaseModel):
    """Account-wide subscription usage, as codex last reported it."""

    plan_type: str | None = None
    primary: RateWindow | None = None
    secondary: RateWindow | None = None


class SessionInfo(BaseModel):
    """One codex session, for the switch list."""

    id: str
    title: str
    started_at: datetime | None = None
    updated_at: datetime | None = None
    git_branch: str | None = None
    cwd: str | None = None


class SessionContext(BaseModel):
    """A snapshot of one session's context usage.

    Not a per-component ``/context`` breakdown - codex does not expose that. These
    are the real axes it does give: the window size, the cumulative token usage,
    and how many turns / tool calls the session has accrued.
    """

    session_id: str
    context_window: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0
    turn_count: int = 0
    tool_call_count: int = 0


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
                title = message.strip()
        if meta is not None and title is not None:
            break
    return meta, title


def list_sessions(codex_home: Path, cwd: str) -> list[SessionInfo]:
    """List this app's codex sessions, newest first.

    Scoped to ``originator == "codex_exec"`` sessions whose ``cwd`` matches the
    server's, which mirrors codex's own default resume filter and keeps unrelated
    codex sessions (other directories, the interactive TUI) out of the list.
    """
    root = _sessions_dir(codex_home)
    if not root.is_dir():
        return []
    sessions: list[SessionInfo] = []
    for path in root.rglob("rollout-*.jsonl"):
        meta, title = _read_head(path)
        if meta is None:
            continue
        if meta.get("originator") != "codex_exec":
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


def read_context(codex_home: Path, session_id: str | None) -> SessionContext | None:
    """The current session's context snapshot, or ``None`` if it cannot be read."""
    if not session_id:
        return None
    path = _find_rollout(codex_home, session_id)
    if path is None:
        return None
    window = 0
    usage: dict[str, Any] | None = None
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
                total = info.get("total_token_usage")
                if isinstance(total, dict):
                    usage = total
    context = SessionContext(
        session_id=session_id,
        context_window=window,
        turn_count=turns,
        tool_call_count=tools,
    )
    if usage is not None:
        context.input_tokens = _int(usage.get("input_tokens"))
        context.cached_input_tokens = _int(usage.get("cached_input_tokens"))
        context.output_tokens = _int(usage.get("output_tokens"))
        context.reasoning_output_tokens = _int(usage.get("reasoning_output_tokens"))
        context.total_tokens = _int(usage.get("total_tokens"))
    return context


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _window(data: Any) -> RateWindow | None:
    if not isinstance(data, dict):
        return None
    used = data.get("used_percent")
    minutes = data.get("window_minutes")
    if used is None or minutes is None:
        return None
    resets = data.get("resets_at")
    return RateWindow(
        used_percent=float(used),
        window_minutes=int(minutes),
        resets_at=int(resets) if isinstance(resets, (int, float)) else None,
    )


def _last_rate_limits(path: Path) -> UsageQuota | None:
    latest: dict[str, Any] | None = None
    for event in _iter_events(path):
        if _event_kind(event) == "token_count":
            rate_limits = _payload(event).get("rate_limits")
            if isinstance(rate_limits, dict):
                latest = rate_limits
    if latest is None:
        return None
    plan = latest.get("plan_type")
    return UsageQuota(
        plan_type=plan if isinstance(plan, str) else None,
        primary=_window(latest.get("primary")),
        secondary=_window(latest.get("secondary")),
    )


def read_usage(codex_home: Path) -> UsageQuota | None:
    """Account-wide usage/quota from the most recent rollout that reported it.

    ``rate_limits`` is account-wide, not session-specific, so the newest rollout
    carrying a ``token_count`` has the freshest figures.
    """
    root = _sessions_dir(codex_home)
    if not root.is_dir():
        return None
    paths = sorted(
        root.rglob("rollout-*.jsonl"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for path in paths:
        quota = _last_rate_limits(path)
        if quota is not None:
            return quota
    return None
