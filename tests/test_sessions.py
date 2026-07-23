"""Tests for codex session introspection.

Integration-style: each test writes fake codex rollout `.jsonl` files into a temp
``CODEX_HOME`` and exercises the real parsing - no codex binary, no network.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from scufris.config import Settings
from scufris.sessions import (
    AGENT_STEERING_PREAMBLE,
    STEERING_PREAMBLE,
    TranscriptMessage,
    delete_session,
    format_fork_seed,
    list_sessions,
    read_context,
    read_transcript,
    read_usage,
    resolve_codex_home,
    strip_steering,
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
    # The app_server backend tags sessions with originator "scufris" (from the
    # initialize clientInfo.name); these must list alongside exec's "codex_exec".
    _write_rollout(
        tmp_path, "ddd-4", cwd="/app", originator="scufris", title="app-server turn"
    )

    sessions = list_sessions(tmp_path, "/app")

    # Both scufris-originated sessions list (app_server + exec); the vscode TUI
    # session and the other-directory one are excluded.
    assert {s.id for s in sessions} == {"aaa-1", "ddd-4"}
    titles = {s.id: s.title for s in sessions}
    assert titles["ddd-4"] == "app-server turn"


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


def test_read_context_uses_last_usage_for_current_fill(tmp_path: Path) -> None:
    # The context bar must reflect the CURRENT occupancy (last request's input),
    # not the cumulative sum across turns (which overcounts past the window).
    token_count = json.dumps(
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "model_context_window": 258400,
                    "total_token_usage": {
                        "input_tokens": 58458,
                        "cached_input_tokens": 40000,
                        "output_tokens": 200,
                        "reasoning_output_tokens": 50,
                        "total_tokens": 58700,
                    },
                    "last_token_usage": {
                        "input_tokens": 15263,
                        "cached_input_tokens": 9000,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                        "total_tokens": 15288,
                    },
                },
            },
        }
    )
    _write_rollout(
        tmp_path, "sess-fill", cwd="/app", turns=2, extra_lines=[token_count]
    )

    context = read_context(tmp_path, "sess-fill")

    assert context is not None
    assert context.input_tokens == 15263  # current fill = last request
    assert context.cached_input_tokens == 9000
    assert context.output_tokens == 200  # cumulative work = total
    assert context.total_tokens == 58700


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


def test_read_transcript_pairs_user_and_assistant(tmp_path: Path) -> None:
    _write_rollout(tmp_path, "sess-t", cwd="/app", title="first?", turns=2)

    messages = read_transcript(tmp_path, "sess-t")

    roles = [m.role for m in messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert messages[0].text == "first?"
    assert messages[1].text == "hi"


def test_strip_steering_removes_preamble_block() -> None:
    assert strip_steering(f"{STEERING_PREAMBLE}\n\nreal question") == "real question"
    # The sub-agent block shares the sentinel markers, so it is cleaned too.
    assert strip_steering(f"{AGENT_STEERING_PREAMBLE}\n\nreal question") == (
        "real question"
    )
    # Idempotent / no-op on plain text.
    assert strip_steering("just a question") == "just a question"
    # Only the leading block is removed; later brackets in the body survive.
    assert strip_steering("plain [scufris-tools] not a block") == (
        "plain [scufris-tools] not a block"
    )


def test_transcript_and_title_hide_the_steering_preamble(tmp_path: Path) -> None:
    # A rollout whose user turns were sent with the injected steering preamble must
    # re-render (and title) as the user's actual text, not the instructions.
    steered = json.dumps(
        {
            "type": "event_msg",
            "payload": {
                "type": "user_message",
                "message": f"{STEERING_PREAMBLE}\n\nwhat is eating my CPU?",
            },
        }
    )
    _write_rollout(tmp_path, "sess-steer", cwd="/app", turns=0, extra_lines=[steered])

    messages = read_transcript(tmp_path, "sess-steer")
    assert messages[0].role == "user"
    assert messages[0].text == "what is eating my CPU?"

    sessions = list_sessions(tmp_path, "/app")
    assert sessions[0].title == "what is eating my CPU?"


def test_read_transcript_carries_event_timestamp(tmp_path: Path) -> None:
    # A rollout event with a top-level timestamp surfaces on the message as `ts`
    # (for the UI clock); an event without one leaves `ts` None.
    stamped = json.dumps(
        {
            "timestamp": "2026-07-19T14:39:39.982Z",
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "stamped question"},
        }
    )
    _write_rollout(tmp_path, "sess-ts", cwd="/app", turns=1, extra_lines=[stamped])

    messages = read_transcript(tmp_path, "sess-ts")

    stamped_msg = messages[-1]
    assert stamped_msg.text == "stamped question"
    assert stamped_msg.ts is not None
    assert stamped_msg.ts.isoformat().startswith("2026-07-19T14:39:39")
    # The turn=1 default messages have no top-level timestamp -> ts stays None.
    assert messages[0].ts is None


def test_read_transcript_attaches_tool_calls_and_usage_to_the_reply(
    tmp_path: Path,
) -> None:
    # A real turn records: user -> commentary agent_message -> mcp_tool_call_end(s)
    # -> final_answer agent_message -> token_count. The tool calls (and the turn's
    # output tokens) must ride on the FINAL assistant message so the chips survive a
    # transcript reload (they only render live otherwise). Reproduces the bug where
    # switching to a past session drops the "ran <tool>" chips.
    day = tmp_path / "sessions" / "2026" / "07" / "20"
    day.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = [
        {
            "type": "session_meta",
            "payload": {"session_id": "sess-tc", "cwd": "/app"},
        },
        {
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "how full are my disks?"},
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "message": "I'll check the disk tool.",
                "phase": "commentary",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "mcp_tool_call_end",
                "call_id": "c1",
                "invocation": {
                    "server": "scufris",
                    "tool": "disk_usage",
                    "arguments": {},
                },
                "result": {"Ok": {}},
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "message": "Your disk is fine.",
                "phase": "final_answer",
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {"output_tokens": 42, "input_tokens": 100}
                },
            },
        },
    ]
    path = day / "rollout-2026-07-20T10-00-00-sess-tc.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    messages = read_transcript(tmp_path, "sess-tc")

    # Commentary is skipped; one assistant message (the final answer) carries the
    # tool call + the turn's output tokens.
    assistants = [m for m in messages if m.role == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].text == "Your disk is fine."
    assert [tc.tool for tc in assistants[0].tool_calls] == ["disk_usage"]
    assert assistants[0].tool_calls[0].server == "scufris"
    assert assistants[0].tool_calls[0].status == "completed"
    assert assistants[0].usage is not None
    assert assistants[0].usage.output_tokens == 42
    # A prior/empty turn's assistant message carries no phantom tool calls.
    assert messages[0].role == "user"
    assert messages[0].tool_calls == []


def test_read_transcript_skips_intermediate_agent_phases(tmp_path: Path) -> None:
    reasoning = json.dumps(
        {
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "message": "thinking out loud",
                "phase": "reasoning",
            },
        }
    )
    _write_rollout(tmp_path, "sess-r", cwd="/app", turns=1, extra_lines=[reasoning])

    texts = [m.text for m in read_transcript(tmp_path, "sess-r")]
    assert "thinking out loud" not in texts


def test_read_transcript_empty_for_unknown(tmp_path: Path) -> None:
    assert read_transcript(tmp_path, None) == []
    assert read_transcript(tmp_path, "nope") == []


def test_delete_session_removes_rollout(tmp_path: Path) -> None:
    path = _write_rollout(tmp_path, "sess-del", cwd="/app")
    assert path.exists()

    assert delete_session(tmp_path, "sess-del") is True
    assert not path.exists()
    assert list_sessions(tmp_path, "/app") == []


def test_delete_session_noop_for_unknown(tmp_path: Path) -> None:
    _write_rollout(tmp_path, "keep-me", cwd="/app")
    assert delete_session(tmp_path, None) is False
    assert delete_session(tmp_path, "does-not-exist") is False
    # The real session is untouched.
    assert [s.id for s in list_sessions(tmp_path, "/app")] == ["keep-me"]


def test_delete_session_logs_the_deletion(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_rollout(tmp_path, "sess-log", cwd="/app")
    with caplog.at_level(logging.INFO, logger="scufris.sessions"):
        delete_session(tmp_path, "sess-log")
    assert any(
        "deleted session sess-log" in record.getMessage() for record in caplog.records
    )


def test_format_fork_seed_includes_context_and_edit() -> None:
    ctx = [
        TranscriptMessage(role="user", text="what is the load?"),
        TranscriptMessage(role="assistant", text="it is 0.5"),
    ]
    seed = format_fork_seed(ctx, "  and the memory?  ")
    assert "User: what is the load?" in seed
    assert "Assistant: it is 0.5" in seed
    assert seed.rstrip().endswith("and the memory?")  # edited text is the last turn


def test_format_fork_seed_no_context_is_just_text() -> None:
    # Forking the very first message is a plain new chat - no context preamble.
    assert format_fork_seed([], "just this") == "just this"


def test_format_fork_seed_caps_context() -> None:
    ctx = [TranscriptMessage(role="user", text=f"m{i}") for i in range(10)]
    seed = format_fork_seed(ctx, "q", max_turns=2)
    assert "m9" in seed and "m8" in seed
    assert "m0" not in seed
