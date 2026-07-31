"""Account-wide quota and the agent's on-disk footprint.

Both answers come from the same rollout tree the rest of the package reads:
``rate_limits`` is account-wide (so the newest rollout that reported it has the
freshest figures), and the footprint is the size and span of the rollouts
themselves.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import RateWindow, UsageQuota
from .rollout import _event_kind, _iter_events, _payload, _sessions_dir


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


class MemoryFootprint(BaseModel):
    """The agent's persistent footprint on disk: its codex session rollouts.

    "Memory" here is deliberately concrete - the rollouts codex keeps, not a
    separate memory system. Surfaced read-only so the operator can see how much
    the agent has accumulated (count, bytes, span).
    """

    session_count: int
    total_bytes: int
    oldest: datetime | None = None
    newest: datetime | None = None


def read_memory_footprint(codex_home: Path) -> MemoryFootprint:
    """Count and size the codex rollouts under ``codex_home``. Never raises.

    Returns an empty footprint (zeros, no dates) when the sessions dir is
    missing, so the endpoint is safe on a box that has never run codex.
    """
    root = _sessions_dir(codex_home)
    empty = MemoryFootprint(session_count=0, total_bytes=0)
    if not root.is_dir():
        return empty
    count = 0
    total = 0
    oldest: datetime | None = None
    newest: datetime | None = None
    for path in root.rglob("rollout-*.jsonl"):
        try:
            stat = path.stat()
        except OSError:
            continue
        count += 1
        total += stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        if oldest is None or mtime < oldest:
            oldest = mtime
        if newest is None or mtime > newest:
            newest = mtime
    return MemoryFootprint(
        session_count=count, total_bytes=total, oldest=oldest, newest=newest
    )
