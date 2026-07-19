"""Tests for codex session introspection.

Integration-style: each test writes fake codex rollout `.jsonl` files into a temp
``CODEX_HOME`` and exercises the real parsing - no codex binary, no network.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from scufris.config import Settings
from scufris.sessions import (
    list_sessions,
    read_context,
    read_usage,
    resolve_codex_home,
)


def _write_rollout(
    home: Path,
    session_id: str,
    *,
    cwd: str,
    originator: str = "codex_exec",
    title: str = "hello",
    turns: int = 1,
    tools: int = 0,
    window: int = 258400,
    used_percent: float = 12.5,
    branch: str = "main",
    with_rate_limits: bool = True,
    extra_lines: list[str] | None = None,
) -> Path:
    """Write a minimal but realistic rollout file and return its path."""
    day = home / "sessions" / "2026" / "07" / "19"
    day.mkdir(parents=True, exist_ok=True)
    path = day / f"rollout-2026-07-19T14-39-30-{session_id}.jsonl"

    events: list[dict[str, Any]] = [
        {
            "type": "session_meta",
            "payload": {
                "session_id": session_id,
                "id": session_id,
                "timestamp": "2026-07-19T14:39:30.556Z",
                "cwd": cwd,
                "originator": originator,
                "git": {"branch": branch},
            },
        }
    ]
    for i in range(turns):
        events.append(
            {
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": title if i == 0 else f"turn {i}",
                },
            }
        )
        events.append(
            {
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": "hi"},
            }
        )
    for _ in range(tools):
        events.append({"type": "event_msg", "payload": {"type": "mcp_tool_call_end"}})
    info = {
        "model_context_window": window,
        "total_token_usage": {
            "input_tokens": 100,
            "cached_input_tokens": 40,
            "output_tokens": 20,
            "reasoning_output_tokens": 5,
            "total_tokens": 120,
        },
    }
    token_count: dict[str, Any] = {"type": "token_count", "info": info}
    if with_rate_limits:
        token_count["rate_limits"] = {
            "plan_type": "plus",
            "primary": {
                "used_percent": used_percent,
                "window_minutes": 10080,
                "resets_at": 1785074524,
            },
            "secondary": None,
        }
    events.append({"type": "event_msg", "payload": token_count})

    lines = [json.dumps(e) for e in events]
    if extra_lines:
        lines.extend(extra_lines)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_resolve_codex_home_prefers_setting(tmp_path: Path) -> None:
    assert resolve_codex_home(Settings(codex_home=tmp_path)) == tmp_path


def test_list_sessions_filters_by_cwd_and_originator(tmp_path: Path) -> None:
    _write_rollout(tmp_path, "aaa-1", cwd="/app", title="first task")
    _write_rollout(tmp_path, "bbb-2", cwd="/elsewhere", title="other dir")
    _write_rollout(tmp_path, "ccc-3", cwd="/app", originator="vscode", title="tui")

    sessions = list_sessions(tmp_path, "/app")

    assert [s.id for s in sessions] == ["aaa-1"]
    assert sessions[0].title == "first task"
    assert sessions[0].git_branch == "main"
    assert sessions[0].cwd == "/app"


def test_list_sessions_sorted_newest_first(tmp_path: Path) -> None:
    old = _write_rollout(tmp_path, "old-1", cwd="/app", title="older")
    new = _write_rollout(tmp_path, "new-2", cwd="/app", title="newer")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    sessions = list_sessions(tmp_path, "/app")
    assert [s.id for s in sessions] == ["new-2", "old-1"]


def test_list_sessions_empty_when_no_dir(tmp_path: Path) -> None:
    assert list_sessions(tmp_path / "nope", "/app") == []


def test_read_context_counts_turns_tools_and_tokens(tmp_path: Path) -> None:
    _write_rollout(tmp_path, "sess-9", cwd="/app", turns=3, tools=2, window=258400)

    context = read_context(tmp_path, "sess-9")

    assert context is not None
    assert context.context_window == 258400
    assert context.turn_count == 3
    assert context.tool_call_count == 2
    assert context.input_tokens == 100
    assert context.cached_input_tokens == 40
    assert context.total_tokens == 120


def test_read_context_none_for_unknown_or_missing(tmp_path: Path) -> None:
    assert read_context(tmp_path, None) is None
    assert read_context(tmp_path, "does-not-exist") is None


def test_read_context_skips_malformed_lines(tmp_path: Path) -> None:
    _write_rollout(
        tmp_path,
        "sess-x",
        cwd="/app",
        turns=1,
        extra_lines=["this is not json", "{bad", ""],
    )
    context = read_context(tmp_path, "sess-x")
    assert context is not None
    assert context.turn_count == 1


def test_read_usage_returns_weekly_window(tmp_path: Path) -> None:
    _write_rollout(tmp_path, "sess-u", cwd="/app", used_percent=34.0)

    usage = read_usage(tmp_path)

    assert usage is not None
    assert usage.plan_type == "plus"
    assert usage.primary is not None
    assert usage.primary.window_minutes == 10080
    assert usage.primary.used_percent == 34.0
    assert usage.primary.resets_at == 1785074524


def test_read_usage_none_when_absent(tmp_path: Path) -> None:
    _write_rollout(tmp_path, "sess-n", cwd="/app", with_rate_limits=False)
    assert read_usage(tmp_path) is None


def test_read_context_treats_session_id_literally(tmp_path: Path) -> None:
    # A glob-metacharacter id must not match a real session's rollout file.
    _write_rollout(tmp_path, "real-1", cwd="/app")
    assert read_context(tmp_path, "*") is None
