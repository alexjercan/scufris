"""The pure formatters: text in, text out, no bot and no transport.

Every function here is called directly with populated, empty, absent and
degraded inputs, because the edge cases are cheapest to pin where nothing has
to be dispatched. Covers the reasoning and tool widgets, the final-answer
tool-summary footer, the Markdown -> MarkdownV2 conversion and its fallback,
and the ``/stats`` and ``/settings`` renderers. Routing those commands through
the bot is ``tests/test_telegram_app.py``'s job.

The emoji constants come from the harness in ``tests/test_telegram.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from conftest import make_fixture_stats
from test_telegram import BRAIN, CHECK, CROSS, WRENCH

import scufris.telegram.render as telegram_render
from scufris.backends import Capability
from scufris.health import AgentHealth, HealthCheck
from scufris.mcp_models import AgentTool
from scufris.sessions import RateWindow, ToolCall, UsageQuota
from scufris.telegram import (
    CAP_EMPTY,
    CAP_UNSUPPORTED,
    HELP_TEXT,
    OrchestratorInfo,
    _command_arg,
    _format_reasoning,
    _format_tool,
    markdown_reply,
    render_health,
    render_reply,
    render_settings_summary,
    render_stats,
    render_tools,
    render_usage,
)
from scufris_host import GpuStats, NetIfRate, SensorGroup, SensorReading

# --- pure widget formatters --------------------------------------------------


def test_format_reasoning_empty_is_header_only() -> None:
    header = f"{BRAIN} <b>Thinking...</b>"
    assert _format_reasoning("") == header
    assert _format_reasoning("   ") == header


def test_format_reasoning_escapes_and_italicises() -> None:
    out = _format_reasoning("a < b & c")
    assert out.startswith(f"{BRAIN} <b>Thinking...</b>")
    # The reasoning body is HTML-escaped and wrapped in italics.
    assert "<i>a &lt; b &amp; c</i>" in out


def test_format_reasoning_tail_windows_long_text() -> None:
    out = _format_reasoning("head" + ("y" * 5000))
    assert len(out) <= 4096  # fits Telegram's message cap
    assert "..." in out  # trimmed marker
    assert "head" not in out  # the tail is kept, the head dropped


def test_format_reasoning_caps_length_after_escaping() -> None:
    # `<` escapes to `&lt;` (4x), so a raw-length trim would blow past 4096; the
    # escape-then-trim keeps the ESCAPED body bounded.
    out = _format_reasoning("HEAD" + ("<" * 5000))
    assert len(out) <= 4096
    assert "HEAD" not in out  # head trimmed, tail kept
    assert "&lt;" in out  # body is escaped


def test_format_tool_ok_and_failed() -> None:
    ok = _format_tool(ToolCall(server="scufris", tool="host_stats", status="success"))
    assert ok == f"{WRENCH} <b>host_stats</b> {CHECK}"
    failed = _format_tool(
        ToolCall(server="scufris", tool="create_agent", status="error")
    )
    assert failed == f"{WRENCH} <b>create_agent</b> {CROSS}"


def test_format_tool_shows_non_default_server() -> None:
    out = _format_tool(ToolCall(server="the-den", tool="today", status="ok"))
    assert "the-den.today" in out


def test_format_tool_escapes_names() -> None:
    out = _format_tool(ToolCall(server="scufris", tool="a<b", status="success"))
    assert "a&lt;b" in out


# --- render_reply (final-answer tool-summary footer) -------------------------


def _tc(tool: str, status: str = "success", server: str = "scufris") -> ToolCall:
    return ToolCall(server=server, tool=tool, status=status)


def test_render_reply_no_tools_is_unchanged() -> None:
    assert render_reply("hello", []) == "hello"


def test_render_reply_appends_tool_footer() -> None:
    rendered = render_reply("done", [_tc("host_stats"), _tc("list_agents")])
    assert rendered == "done\n\ntools: host_stats, list_agents"


def test_render_reply_counts_repeated_tools_in_call_order() -> None:
    rendered = render_reply(
        "ok",
        [_tc("list_agents"), _tc("host_stats"), _tc("list_agents")],
    )
    # First-seen order, with a count for the repeated tool.
    assert rendered == "ok\n\ntools: list_agents x2, host_stats"


def test_render_reply_marks_failed_calls() -> None:
    rendered = render_reply("oops", [_tc("create_agent", status="error")])
    assert rendered == "oops\n\ntools: create_agent (failed)"


def test_render_reply_empty_text_with_tools_is_footer_only() -> None:
    # A tools-only turn must still yield a non-empty body so the caller's
    # empty-reply coalesce does not swallow it.
    assert render_reply("", [_tc("host_stats")]) == "tools: host_stats"


# --- markdown_reply (final-answer -> Telegram MarkdownV2) ---------------------

# A model answer exercising every transform the wrapper must handle: heading,
# emphasis, inline code, link, bulleted + numbered lists, and a GFM table.
_MD_ANSWER = (
    "# Report\n\n"
    "Here are the **results** with a [link](https://example.com) and `inline`.\n\n"
    "- first item\n"
    "- second item\n\n"
    "1. step one\n"
    "2. step two\n\n"
    "| Name | Score |\n"
    "| ---- | ----- |\n"
    "| Alice | 42 |\n"
    "| Bob | 7 |\n"
)


def test_markdown_reply_transforms_heading_list_and_table() -> None:
    body = markdown_reply(_MD_ANSWER, [])
    # Heading is no longer a literal "# " line; it renders bold.
    assert "# Report" not in body
    assert "*Report*" in body
    # A GFM table has no Telegram equivalent, so it becomes a monospace code
    # block (fenced) with the cell values aligned inside it.
    assert "```" in body
    assert "Alice" in body and "42" in body and "Bob" in body
    # Bullets render with a real bullet glyph, not a literal leading "- ".
    assert "⦁" in body  # bullet
    assert "first item" in body and "second item" in body
    # Emphasis / inline code / link survive the conversion.
    assert "*results*" in body
    assert "`inline`" in body
    assert "[link](https://example.com)" in body


def test_markdown_reply_escapes_markdownv2_specials() -> None:
    # A bare "." or "!" is a MarkdownV2 special that would 400 the send; the
    # wrapper must escape them (backslash-escaped) so the body is safe.
    body = markdown_reply("Version 1.0 released!", [])
    assert "1\\.0" in body
    assert "released\\!" in body


def test_markdown_reply_preserves_and_escapes_tool_footer() -> None:
    # The ASCII tools footer is carried through; the underscores in tool names
    # are MarkdownV2 specials and come out backslash-escaped (rendered as "_").
    body = markdown_reply("done", [_tc("host_stats"), _tc("list_agents")])
    assert body.endswith("tools: host\\_stats, list\\_agents")


def test_markdown_reply_empty_answer_is_empty() -> None:
    # No text and no tools -> empty body, so the caller keeps its empty-reply
    # coalesce (the fixed EMPTY_REPLY notice) instead of sending "".
    assert markdown_reply("", []) == ""


def test_markdown_reply_converter_failure_falls_back_to_plain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A converter bug must never lose the reply: on any exception the wrapper
    # returns the raw render_reply body (still deliverable as plain text).
    def boom(*_a: Any, **_k: Any) -> str:
        raise RuntimeError("converter exploded")

    monkeypatch.setattr(telegram_render.telegramify_markdown, "markdownify", boom)
    body = markdown_reply("hello world", [_tc("host_stats")])
    assert body == "hello world\n\ntools: host_stats"


WARN = "\N{WARNING SIGN}"


def _fake_health() -> AgentHealth:
    return AgentHealth(
        scufris_version="0.1.0",
        backend="codex",
        backend_version="0.20.0",
        session_count=3,
        last_session=datetime(2026, 7, 20, tzinfo=timezone.utc),
        checks=[
            HealthCheck(name="agent", status="ok", detail="enabled (backend codex)"),
            HealthCheck(
                name="codex auth",
                status="warn",
                detail="unknown",
                hint="run `codex login`",
            ),
        ],
    )


def _fake_usage() -> UsageQuota:
    return UsageQuota(
        plan_type="pro",
        primary=RateWindow(
            used_percent=42.0, window_minutes=10080, resets_at=1795000000
        ),
        secondary=None,
    )


def _fake_tools() -> list[AgentTool]:
    return [
        AgentTool(name="host_stats", description="host metrics", server="scufris"),
        AgentTool(
            name="processes",
            description="process list",
            server="scufris",
            enabled=False,
        ),
        AgentTool(name="journal_today", description="today's journal", server="den"),
    ]


def _fake_info(
    backend: str = "codex", quota: Capability[UsageQuota] | None = None
) -> OrchestratorInfo:
    return OrchestratorInfo(
        backend=backend,
        model="gpt-5.5",
        auth_mode="chatgpt",
        enabled=True,
        permission_mode="auto",
        quota=Capability.read(_fake_usage()) if quota is None else quota,
    )


def test_command_arg_extracts_subcommand() -> None:
    assert _command_arg("/settings health") == "health"
    assert _command_arg("/settings HEALTH") == "health"
    assert _command_arg("/settings") == ""
    assert _command_arg("/settings@mybot usage extra") == "usage"
    assert _command_arg("plain text") == "text"


def test_render_stats_is_compact_health_snapshot() -> None:
    stats = make_fixture_stats().model_copy(
        update={
            "net_interfaces": [
                NetIfRate(
                    name="eth0",
                    sent_per_sec=1024**2 * 1.2,
                    recv_per_sec=1024**2 * 0.3,
                )
            ],
            "temps": [
                SensorGroup(
                    chip="coretemp",
                    readings=[SensorReading(label="Package", current=48.0)],
                )
            ],
            "gpus": [
                GpuStats(
                    name="rtx",
                    util_percent=5.0,
                    mem_used_mb=1024,
                    mem_total_mb=8192,
                    mem_percent=12.5,
                    temp_c=39.0,
                    power_w=30.0,
                    power_limit_w=250.0,
                    clock_sm_mhz=1500,
                    clock_mem_mhz=7000,
                )
            ],
            "process_count": 312,
        }
    )
    body = render_stats(stats)
    assert "Host stats" in body
    assert "host: testbox  up 20m" in body
    assert "CPU 12%  load 0.10 / 0.20 / 0.30" in body
    assert "MEM 40%" in body and "swap 25%" in body
    assert "disk / 20%" in body
    assert "net up 1.2 / down 0.3 MB/s" in body
    assert "temp 48C (Package)  procs 312" in body
    assert "GPU 0 5%  39C  1.0/8.0G" in body


def test_render_stats_omits_absent_sections() -> None:
    # The bare fixture has no net interfaces / temps / gpus: those lines vanish
    # rather than render empty, and the tail falls back to just the process count.
    body = render_stats(make_fixture_stats())
    assert "net " not in body
    assert "temp " not in body
    assert "GPU " not in body
    assert "procs 0" in body


def test_render_health_marks_each_check() -> None:
    body = render_health(_fake_health())
    assert "scufris 0.1.0  backend codex 0.20.0" in body
    assert "sessions 3  last 2026-07-20" in body
    assert f"{CHECK} agent: enabled (backend codex)" in body
    assert f"{WARN} codex auth: unknown" in body
    # The hint shows on the warn check, and its backticks are scrubbed so they
    # cannot break the surrounding code fence.
    assert "hint: run 'codex login'" in body
    assert "run `codex login`" not in body


def test_render_health_omits_the_session_line_without_a_reading() -> None:
    # session_count None means no reading was taken: the whole line goes, rather
    # than rendering "sessions None".
    health = _fake_health().model_copy(
        update={"session_count": None, "last_session": None}
    )
    body = render_health(health)
    assert "sessions" not in body
    assert "scufris 0.1.0  backend codex 0.20.0" in body
    assert f"{CHECK} agent: enabled (backend codex)" in body
    assert f"{WARN} codex auth: unknown" in body


def test_render_usage_tells_the_three_capability_states_apart() -> None:
    body = render_usage(_fake_info())
    assert "plan: pro" in body
    assert "primary (weekly): 42% used" in body
    assert "resets 2026-" in body
    # A reader that ran and found nothing is not a backend with no reader, and
    # the unsupported reading names the backend that cannot answer.
    empty = render_usage(_fake_info(quota=Capability.read(None)))
    assert CAP_EMPTY in empty
    unsupported = render_usage(_fake_info("claude", Capability.unsupported()))
    assert CAP_UNSUPPORTED.format(backend="claude") in unsupported
    assert CAP_EMPTY not in unsupported


def test_render_tools_groups_by_server_and_flags_disabled() -> None:
    body = render_tools(_fake_tools())
    assert "2/3 tools enabled" in body
    assert "[den] (1)" in body
    assert "[scufris] (2)" in body
    assert "- processes  (disabled)" in body
    assert "- host_stats" in body
    assert render_tools([]).endswith("no tools available\n```")


def test_render_settings_summary_rolls_up_worst_status() -> None:
    body = render_settings_summary(_fake_info(), _fake_health(), _fake_tools())
    assert "backend: codex  model: gpt-5.5" in body
    assert "auth: chatgpt  enabled: yes" in body
    assert "permission: auto" in body
    assert "tools: 2/3 enabled" in body
    assert "usage: 42% (weekly)" in body
    # A warn check makes the rolled-up health warn, not ok.
    assert f"health: {WARN} warn" in body
    assert "Subcommands: /settings health | usage | tools" in body


def test_render_settings_summary_carries_the_capability_reading() -> None:
    """The summary's usage line says the same three things `/settings usage` does,
    so the two bodies cannot disagree about the same envelope."""
    empty = render_settings_summary(
        _fake_info(quota=Capability.read(None)), _fake_health(), _fake_tools()
    )
    assert f"usage: {CAP_EMPTY}" in empty
    unsupported = render_settings_summary(
        _fake_info("opencode", Capability.unsupported()), _fake_health(), _fake_tools()
    )
    assert f"usage: {CAP_UNSUPPORTED.format(backend='opencode')}" in unsupported
    assert "%" not in unsupported
    # A quota that only fills the secondary window is still a reading: the
    # summary must print it rather than claim nothing was reported.
    secondary_only = UsageQuota(
        plan_type="pro",
        primary=None,
        secondary=RateWindow(
            used_percent=17.0, window_minutes=1440, resets_at=1795000000
        ),
    )
    summary = render_settings_summary(
        _fake_info(quota=Capability.read(secondary_only)),
        _fake_health(),
        _fake_tools(),
    )
    assert "usage: 17% (daily)" in summary
    assert CAP_EMPTY not in summary


def test_help_text_lists_settings_and_stats() -> None:
    assert "/settings" in HELP_TEXT
    assert "/stats" in HELP_TEXT
