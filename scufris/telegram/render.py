"""Pure renderers: a domain model in, a Telegram message body out.

Three bodies leave this module. The turn's FINAL answer is the model's
GitHub-flavoured markdown converted to Telegram MarkdownV2 (``markdown_reply``
-> ``telegramify_markdown``): headings become bold, lists become bullets, and a
table becomes an aligned monospace code block, so the user sees formatted output
rather than raw ``#``/``|``/``-``. Because that reintroduces the parse-error risk
the old plain-text answer avoided, ``render_reply`` also yields the raw body, and
the send path falls back to it (see `api.BotApi.send_reply`).

The live "thinking" and tool-call widgets are HTML, using emoji - a deliberate
exception to the repo's ASCII-only convention, scoped to the Telegram rendered
surface (the emoji are ``\\N{...}`` escapes so the SOURCE stays ASCII).

The read-only `/settings` and `/stats` bodies are a bold title over a fenced code
block, whose monospace and preserved newlines keep an aligned key/value read-out
intact; ``settings_markdown`` converts them with the same fallback. Model or probe
text landing inside a fence is scrubbed of backticks so a stray one cannot break
it.

The host-approval message and its keyboards are here too: the body is the SHARED
`host_actions.render_action`, never a Telegram-shaped paraphrase.
"""

from __future__ import annotations

import html
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import telegramify_markdown

from scufris_host import HostStats
from scufris_hostctl import HostActionRecord, render_action

from ..health import AgentHealth
from ..mcp_models import AgentTool
from ..sessions import ToolCall
from .contracts import OrchestratorInfo
from .text import (
    CAP_EMPTY,
    CAP_UNSUPPORTED,
    CB_ABORT,
    CB_APPROVE,
    CB_CONFIRM,
    CB_DENY,
    MAX_MESSAGE,
)

logger = logging.getLogger(__name__)

# Bound on the ESCAPED reasoning body, kept well under Telegram's 4096-char cap;
# the TAIL is kept (the current train of thought) with a leading ellipsis when
# trimmed. Bounding the escaped length (not the raw) is what makes the cap airtight.
_REASONING_MAX = 3500

# Tool statuses the backends report for a successful call (anything else is
# surfaced as failed). codex/claude report "success"; the mock/other paths may
# use "ok". Shared by the final-answer footer and the live tool widget.
_OK_STATUSES = frozenset({"ok", "success", "completed", "done"})

# Emoji for the widgets, as \N{...} escapes so the SOURCE stays ASCII (the repo's
# ASCII-only convention) while the RENDERED Telegram message shows the glyph.
_EMOJI_THINKING = "\N{BRAIN}"
_EMOJI_TOOL = "\N{WRENCH}"
_EMOJI_OK = "\N{HEAVY CHECK MARK}"
_EMOJI_WARN = "\N{WARNING SIGN}"
_EMOJI_FAIL = "\N{CROSS MARK}"

# Replaces the preview lines trimmed out of an over-long approval message; see
# `render_approval` for why the PREVIEW is what gets shortened.
_ELIDED = "  [...] {n} more preview lines - read them on the dashboard's /host/ page"


def render_reply(text: str, tool_calls: Sequence[ToolCall]) -> str:
    """Render an orchestrator turn's FINAL answer into the single Telegram message.

    The reply text is returned unchanged when the turn made no tool calls. When
    it did, a compact ASCII footer line is appended (blank line + ``tools: ...``)
    listing the unique tool names in call order, with ``xN`` for a repeated tool
    and ``(failed)`` when any call of that tool did not report success. ASCII only
    (no emoji/typographic chars) so the plain-text ``sendMessage`` stays clean and
    the model's own text is never re-interpreted by an HTML ``parse_mode``.

    Empty ``text`` with tool calls yields a footer-only body (still non-empty, so
    the caller's empty-reply coalesce does not swallow a tools-only turn)."""
    if not tool_calls:
        return text
    order: list[str] = []
    counts: dict[str, int] = {}
    failed: set[str] = set()
    for call in tool_calls:
        if call.tool not in counts:
            order.append(call.tool)
            counts[call.tool] = 0
        counts[call.tool] += 1
        if call.status.lower() not in _OK_STATUSES:
            failed.add(call.tool)
    parts: list[str] = []
    for tool in order:
        label = tool
        if counts[tool] > 1:
            label += f" x{counts[tool]}"
        if tool in failed:
            label += " (failed)"
        parts.append(label)
    footer = "tools: " + ", ".join(parts)
    return f"{text}\n\n{footer}" if text else footer


def markdown_reply(text: str, tool_calls: Sequence[ToolCall]) -> str:
    """Render an orchestrator turn's FINAL answer as a Telegram MarkdownV2 body.

    Builds the same combined body as ``render_reply`` (model text + optional
    ASCII ``tools:`` footer) and converts it from GitHub-flavoured markdown to
    Telegram MarkdownV2 via ``telegramify_markdown.markdownify``: a heading
    becomes bold, a list becomes bullets, a GFM table becomes an aligned
    monospace code block (Telegram has no table primitive), and every MarkdownV2
    special char is backslash-escaped. The result MUST be sent with
    ``parse_mode=MarkdownV2`` (see ``api.BotApi.send_reply``).

    Robustness is load-bearing: this REPLACES the pre-markdown "plain text never
    400s" guarantee, so any converter failure falls back to the raw
    ``render_reply`` body (still deliverable as plain text). An empty answer
    yields "" so the caller keeps its empty-reply coalesce (the fixed
    ``EMPTY_REPLY`` notice) instead of formatting an empty string."""
    plain = render_reply(text, tool_calls)
    if not plain:
        return ""
    try:
        return telegramify_markdown.markdownify(plain)
    except Exception:
        logger.warning(
            "telegram markdown conversion failed; sending plain text", exc_info=True
        )
        return plain


def _format_reasoning(buf: str) -> str:
    """The live "thinking" message body for the accumulated reasoning ``buf``.

    HTML (a bold header + the reasoning in italics), so it must be sent with
    ``parse_mode=HTML``. The reasoning is HTML-escaped (it is model text, may hold
    ``<``/``&``) and the ESCAPED body is tail-windowed to ``_REASONING_MAX`` chars -
    the most recent thought is what matters and the whole must fit Telegram's
    4096-char cap. Escaping happens BEFORE the trim on purpose: ``html.escape`` can
    expand one char into up to ~6 (``&amp;``), so trimming the raw text would not
    actually bound the message length. Cutting inside an escaped entity only ever
    drops its leading ``&...`` (the TAIL is kept), so no bare ``&`` survives."""
    text = buf.strip()
    if not text:
        return f"{_EMOJI_THINKING} <b>Thinking...</b>"
    body = html.escape(text)
    if len(body) > _REASONING_MAX:
        body = "..." + body[-_REASONING_MAX:]
    return f"{_EMOJI_THINKING} <b>Thinking...</b>\n\n<i>{body}</i>"


def _format_tool(call: ToolCall) -> str:
    """One tool-call widget message: wrench + tool name + a status check/cross.

    HTML (``parse_mode=HTML``); the tool/server names are HTML-escaped. The server
    is shown only when it is not the default built-in ``scufris`` server, so the
    common case stays terse."""
    ok = call.status.lower() in _OK_STATUSES
    mark = _EMOJI_OK if ok else _EMOJI_FAIL
    name = html.escape(call.tool)
    if call.server and call.server != "scufris":
        name = f"{html.escape(call.server)}.{name}"
    return f"{_EMOJI_TOOL} <b>{name}</b> {mark}"


# --- Read-only /settings + /stats rendering ------------------------------------


def _scrub(text: str) -> str:
    """Neutralise backticks in model/probe text destined for a code fence."""
    return text.replace("`", "'")


def _fenced(title: str, body: str) -> str:
    """A bold ``title`` over ``body`` in a fenced code block (aligned monospace)."""
    return f"**{title}**\n```\n{body}\n```"


def _gib(num_bytes: int) -> str:
    """Bytes as GiB with one decimal (for the memory line)."""
    return f"{num_bytes / 1024**3:.1f}"


def _mib_per_sec(bytes_per_sec: float) -> str:
    """A byte/s rate as MiB/s with one decimal (for the net line)."""
    return f"{bytes_per_sec / 1024**2:.1f}"


def _fmt_uptime(seconds: float) -> str:
    """A compact uptime string: ``3d 4h`` / ``4h 12m`` / ``12m``."""
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _fmt_window(minutes: int) -> str:
    """A rate-limit window's human label (codex reports a weekly primary)."""
    return {60: "hourly", 1440: "daily", 10080: "weekly"}.get(minutes, f"{minutes}m")


def _fmt_ts(unix_seconds: int) -> str:
    """A unix reset timestamp as an absolute UTC minute (deterministic)."""
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def _health_mark(status: str) -> str:
    """The check/warn/cross glyph for a health status (ok|warn|error)."""
    return {"ok": _EMOJI_OK, "warn": _EMOJI_WARN, "error": _EMOJI_FAIL}.get(status, "?")


def _worst_status(statuses: Sequence[str]) -> str:
    """The most severe status in a set of checks: error > warn > ok."""
    values = {s for s in statuses}
    if "error" in values:
        return "error"
    if "warn" in values:
        return "warn"
    return "ok"


def _hottest_temp(stats: HostStats) -> tuple[str, float] | None:
    """The single hottest sensor reading (label, celsius), or None if no sensors."""
    best: tuple[str, float] | None = None
    for group in stats.temps:
        for reading in group.readings:
            if best is None or reading.current > best[1]:
                best = (reading.label or group.chip, reading.current)
    return best


def render_stats(stats: HostStats) -> str:
    """A COMPACT host-health snapshot: one tidy block answering "is the box ok".

    Host + uptime, CPU% + load average, memory %/used/total + swap %, disk % per
    mount, aggregate net up/down rate, the hottest sensor + process count, and one
    line per GPU when present. Optimised for a glance, not a full metrics dump."""
    lines = [f"host: {_scrub(stats.hostname)}  up {_fmt_uptime(stats.uptime_seconds)}"]
    load = stats.load_avg
    lines.append(
        f"CPU {stats.cpu_percent:.0f}%  "
        f"load {load[0]:.2f} / {load[1]:.2f} / {load[2]:.2f}"
    )
    mem = stats.mem
    lines.append(
        f"MEM {mem.percent:.0f}% ({_gib(mem.used)}/{_gib(mem.total)}G)  "
        f"swap {stats.swap.percent:.0f}%"
    )
    if stats.disks:
        disk = "  ".join(
            f"{_scrub(d.mountpoint)} {d.percent:.0f}%" for d in stats.disks
        )
        lines.append(f"disk {disk}")
    if stats.net_interfaces:
        up = sum(n.sent_per_sec for n in stats.net_interfaces)
        down = sum(n.recv_per_sec for n in stats.net_interfaces)
        lines.append(f"net up {_mib_per_sec(up)} / down {_mib_per_sec(down)} MB/s")
    hot = _hottest_temp(stats)
    tail = f"procs {stats.process_count}"
    if hot is not None:
        tail = f"temp {hot[1]:.0f}C ({_scrub(hot[0])})  " + tail
    lines.append(tail)
    for idx, gpu in enumerate(stats.gpus):
        lines.append(
            f"GPU {idx} {gpu.util_percent:.0f}%  {gpu.temp_c:.0f}C  "
            f"{gpu.mem_used_mb / 1024:.1f}/{gpu.mem_total_mb / 1024:.1f}G"
        )
    return _fenced("Host stats", "\n".join(lines))


def render_health(health: AgentHealth) -> str:
    """The Health card: scufris + backend version, session summary, and each
    diagnostic check as a glyph + name + detail, with the hint on a warn/error."""
    head = f"scufris {health.scufris_version}  backend {health.backend}"
    if health.backend_version:
        head += f" {_scrub(health.backend_version)}"
    lines = [head]
    # No reading taken (the backend has no session reader) -> no session line.
    if health.session_count is not None:
        session_line = f"sessions {health.session_count}"
        if health.last_session is not None:
            session_line += f"  last {health.last_session:%Y-%m-%d}"
        lines.append(session_line)
    lines.append("")
    for check in health.checks:
        lines.append(
            f"{_health_mark(check.status)} {_scrub(check.name)}: {_scrub(check.detail)}"
        )
        if check.hint and check.status != "ok":
            lines.append(f"   hint: {_scrub(check.hint)}")
    return _fenced("Health", "\n".join(lines))


def _quota_reading(info: OrchestratorInfo) -> str:
    """What to print in place of a quota the caller found no measurement in.

    Two of the three states of `Capability`, in the vocabulary `text` fixes and
    the web duplicates: a backend with no quota reader is not a reader that found
    nothing, so the two must not read alike. The third state - a measurement - is
    the caller's to render."""
    if not info.quota.supported:
        return CAP_UNSUPPORTED.format(backend=_scrub(info.backend))
    return CAP_EMPTY


def render_usage(info: OrchestratorInfo) -> str:
    """The account usage/quota: plan + each rate window's used % and reset time.

    Takes the whole `OrchestratorInfo` rather than a bare quota so an unsupported
    backend can name itself, and so both `/settings` bodies come from the one
    service call `info()` already makes."""
    usage = info.quota.value
    if usage is None or (usage.primary is None and usage.secondary is None):
        return _fenced("Usage", _quota_reading(info))
    windows = [
        (label, window)
        for label, window in (
            ("primary", usage.primary),
            ("secondary", usage.secondary),
        )
        if window is not None
    ]
    lines: list[str] = []
    if usage.plan_type:
        lines.append(f"plan: {_scrub(usage.plan_type)}")
    for label, window in windows:
        line = (
            f"{label} ({_fmt_window(window.window_minutes)}): "
            f"{window.used_percent:.0f}% used"
        )
        if window.resets_at:
            line += f", resets {_fmt_ts(window.resets_at)}"
        lines.append(line)
    return _fenced("Usage", "\n".join(lines))


def render_tools(tools: Sequence[AgentTool]) -> str:
    """The orchestrator tool catalog: enabled/total count then tools grouped by
    server (scufris/den), disabled or unavailable tools flagged."""
    if not tools:
        return _fenced("Tools", "no tools available")
    by_server: dict[str, list[AgentTool]] = {}
    for tool in tools:
        by_server.setdefault(tool.server, []).append(tool)
    enabled = sum(1 for tool in tools if tool.enabled)
    lines = [f"{enabled}/{len(tools)} tools enabled"]
    for server in sorted(by_server):
        server_tools = by_server[server]
        lines.append("")
        lines.append(f"[{_scrub(server)}] ({len(server_tools)})")
        for tool in sorted(server_tools, key=lambda t: t.name):
            flag = "" if tool.enabled else "  (disabled)"
            if not tool.available:
                flag += "  (unavailable)"
            lines.append(f"- {_scrub(tool.name)}{flag}")
    return _fenced("Tools", "\n".join(lines))


def render_settings_summary(
    info: OrchestratorInfo,
    health: AgentHealth,
    tools: Sequence[AgentTool],
) -> str:
    """The `/settings` summary: the config line, tool count, primary usage %, and
    the worst health status - the same at-a-glance view the web dashboard opens
    with. A trailing line points at the detail subcommands."""
    worst = _worst_status([check.status for check in health.checks])
    enabled = sum(1 for tool in tools if tool.enabled)
    usage = info.quota.value
    window = (usage.primary or usage.secondary) if usage is not None else None
    usage_line = (
        f"{window.used_percent:.0f}% ({_fmt_window(window.window_minutes)})"
        if window is not None
        else _quota_reading(info)
    )
    lines = [
        f"backend: {_scrub(info.backend)}  model: {_scrub(info.model)}",
        f"auth: {info.auth_mode or 'none'}  enabled: {'yes' if info.enabled else 'no'}",
        f"permission: {_scrub(info.permission_mode)}",
        f"tools: {enabled}/{len(tools)} enabled",
        f"usage: {usage_line}",
        f"health: {_health_mark(worst)} {worst}",
    ]
    body = _fenced("Settings", "\n".join(lines))
    return f"{body}\nSubcommands: /settings health | usage | tools"


def settings_markdown(body: str) -> str:
    """Convert a `/settings`|`/stats` GFM body to MarkdownV2, falling back to the
    raw body on any converter error (mirrors ``markdown_reply``'s guarantee that a
    reply is never dropped by formatting)."""
    try:
        return telegramify_markdown.markdownify(body)
    except Exception:
        logger.warning(
            "telegram settings markdown conversion failed; sending plain text",
            exc_info=True,
        )
        return body


def render_approval(record: HostActionRecord) -> str:
    """One host action as the message the operator decides from.

    Deliberately `host_actions.render_action` - the SAME text the proposing agent
    is shown and `examples/host_action.py` prints - rather than a Telegram-shaped
    paraphrase. Two surfaces over one decision must not describe it differently,
    and the label saying whether the preview is a simulation or a statement of
    current state, plus the undo line, are the parts a paraphrase drops
    (`share-one-renderer-so-two-surfaces-cannot-drift`).

    Sent WITHOUT a parse mode: it is preformatted plain text holding command lines,
    store paths and journal output, so any markdown/HTML mode would either mangle it
    or reject it.

    Over Telegram's limit, the PREVIEW LINES are what gets shortened - not the tail.
    The undo line and the result sit at the END, so trimming the tail would cost the
    operator the two sentences that matter most, on the class of action most likely
    to be long (an R3 activation's preview IS a closure diff).
    """
    body = render_action(record)
    if len(body) <= MAX_MESSAGE:
        return body
    # Shorten the preview on a COPY, then re-render: one renderer, still, and the
    # head, the commands, the undo line and the result all survive by construction.
    lines = list(record.proposal.preview.lines)
    keep = len(lines)
    while keep > 0:
        keep -= 1
        trimmed = record.model_copy(deep=True)
        trimmed.proposal.preview.lines = lines[:keep] + [
            _ELIDED.format(n=len(lines) - keep)
        ]
        body = render_action(trimmed)
        if len(body) <= MAX_MESSAGE:
            return body
    # Even with no preview at all it does not fit (a pathological summary or a
    # hundred commands): cut the tail as the last resort, which is better than a
    # message Telegram refuses outright.
    return body[:MAX_MESSAGE]


def approval_keyboard(record: HostActionRecord) -> dict[str, Any] | None:
    """The inline keyboard for a PENDING action, or None when there is nothing to
    decide (already decided, expired, drifted - the app says which).

    The one-way case gets a first tap that only ARMS the decision: it says what
    cannot be undone and offers a differently-worded second button, which is the
    proportionate confirmation this surface owes an action that destroys something.
    The ordinary case is one tap, with its undo sentence already in the message.
    """
    if record.decision != "pending":
        return None
    action_id = record.proposal.id
    approve_label = (
        "Approve - CANNOT BE UNDONE"
        if record.confirmation.style == "one_way"
        else "Approve"
    )
    return {
        "inline_keyboard": [
            [
                {"text": approve_label, "callback_data": f"{CB_APPROVE}:{action_id}"},
                {"text": "Deny", "callback_data": f"{CB_DENY}:{action_id}"},
            ]
        ]
    }


def confirm_keyboard(record: HostActionRecord) -> dict[str, Any]:
    """The armed keyboard for a one-way action: the second, explicit tap."""
    action_id = record.proposal.id
    return {
        "inline_keyboard": [
            [
                {
                    "text": f"Yes - {record.proposal.kind}, permanently",
                    "callback_data": f"{CB_CONFIRM}:{action_id}",
                },
                {"text": "Back", "callback_data": f"{CB_ABORT}:{action_id}"},
            ]
        ]
    }
