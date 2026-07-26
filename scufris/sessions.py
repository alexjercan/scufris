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
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from .config import Settings

# The agent prepends one of these steering blocks to each turn's prompt because
# codex obeys only the turn prompt for tool CHOICE: softer channels (tool
# descriptions, an instructions file, an AGENTS.md) were probed and DO NOT steer it
# (lesson codex-tool-choice-only-steers-via-the-turn-prompt). Both blocks share the
# same sentinel wrapping so ``strip_steering`` cleans either one out of titles and
# re-rendered transcripts, and the user never sees them.
#
# ``STEERING_PREAMBLE`` rides ONLY the orchestrator's turns (the host-tools scufris
# server is orchestrator-only; see agent._steer). It is one sentinel block carrying
# three orchestrator-only clauses: the host-tools clause (prefer the curated tools
# over shell), the comms clause (poll for blocked sub-agents and answer them), and
# the journal clause (record plain-language journal/macros facts through the den
# tools - "log that I had 2 eggs" -> macros_lookup then journal_add_macros).
# They share ONE block on purpose - ``strip_steering`` removes only the single
# leading block, so a second sentinel-wrapped block would survive uncleaned.
# ``AGENT_STEERING_PREAMBLE`` rides a tool-having codex SUB-AGENT's turns, teaching
# it to signal when blocked via the ``request_input`` callback it holds (its only
# scufris tool). agent.py imports both; sessions.py owns the format and its inverse
# so they cannot drift.
_STEER_OPEN = "[scufris-tools]"
_STEER_CLOSE = "[/scufris-tools]"
_HOST_TOOLS_CLAUSE = (
    'This host runs a "scufris" MCP server with curated tools: host_stats, '
    "disk_usage, list_processes. For questions about "
    "this host (CPU, memory, swap, disks, network, GPUs, load, processes, uptime) "
    "call host_stats / disk_usage / list_processes FIRST and answer from them; do "
    "NOT use shell commands like uname, lscpu, df, free, top, ps, nvidia-smi or read "
    "/proc for information those tools provide. "
    "Only fall back to the shell when no scufris tool covers it."
)
# The comms clause closes the request_input round-trip on the poll path (auto_wake
# is off by default), so a sub-agent that signalled does not wait forever. The tools
# are orchestrator-only (message_agent / pending_agents / acknowledge); the
# sub-agent side is steered by AGENT_STEERING_PREAMBLE to call request_input.
_COMMS_CLAUSE = (
    "You may have launched sub-agents. At the END of a turn, call pending_agents to "
    "find any that need you (they called request_input and are waiting, or errored); "
    "for each, answer it with message_agent(agent_id, message) - this resumes its "
    "session with your reply - and then call acknowledge(agent_id) so it stops "
    "pending. Do not leave a waiting sub-agent unanswered."
)
# The journal clause steers the orchestrator to the operator's the-den journal +
# macros tools for plain-language journal facts. Like the host-tools clause it must
# ride the TURN PROMPT: codex ignores tool descriptions for tool choice
# (codex-tool-choice-only-steers-via-the-turn-prompt), so "log that I had 2 eggs"
# does not reach the food tools on the strength of their docstrings alone. Every tool
# name and signature here is copied verbatim from mcp_server.py
# (ground-steering-text-in-the-real-tool-signatures): the meal chain is exact -
# macros_lookup("egg 2p") returns "egg 2pc,12,0,10", which IS the CSV row
# journal_add_macros(row) accepts, so the two chain with no reshaping. These tools are
# orchestrator-only (apply_role strips them for a sub-agent), so this clause rides
# only the orchestrator's STEERING_PREAMBLE.
_JOURNAL_CLAUSE = (
    "The operator keeps a daily journal (the-den) reachable through scufris tools: "
    "journal_show reads the day (tasks, habits, macros, weight); journal_add_task, "
    "journal_complete_task, journal_remove_task, journal_toggle_habit, "
    "journal_log_weight, journal_add_note and journal_add_macros write to it. When "
    "the operator states a journal fact in plain language - ate a food, finished or "
    "wants a task, did a habit, a body weight, a note to jot - USE these tools to "
    "record it; do not answer from memory or edit journal files. To LOG A MEAL "
    '("I ate 2 eggs" / "log that I had 2 eggs"), first call macros_lookup(query) '
    'with the food plus amount (e.g. "egg 2p"); it returns a '
    '"<food> <amount><unit>,<protein>,<carbs>,<fat>" row that is EXACTLY what '
    "journal_add_macros(row) accepts, so pass that row straight through to log it. "
    "If the food is unknown, use macros_search(query) to find the name, or "
    "macros_add_food(row) to add it, before macros_lookup."
)
STEERING_PREAMBLE = (
    f"{_STEER_OPEN}\n{_HOST_TOOLS_CLAUSE}\n{_COMMS_CLAUSE}\n"
    f"{_JOURNAL_CLAUSE}\n{_STEER_CLOSE}"
)
AGENT_STEERING_PREAMBLE = (
    f"{_STEER_OPEN}\n"
    "If you are blocked or need a decision or approval you cannot safely make "
    "yourself, call request_input(question) with a clear, specific question and "
    "STOP; do not guess and do not stop silently - the orchestrator will answer and "
    "resume you with the reply in context.\n"
    f"{_STEER_CLOSE}"
)

_STEER_RE = re.compile(
    re.escape(_STEER_OPEN) + r".*?" + re.escape(_STEER_CLOSE) + r"\s*",
    re.DOTALL,
)


def strip_steering(text: str) -> str:
    """Remove a leading scufris steering block (see ``STEERING_PREAMBLE``) from a
    recorded user message, so titles and re-rendered transcripts show only what the
    user actually typed. A no-op when the text has no steering block."""
    return _STEER_RE.sub("", text, count=1).lstrip()


logger = logging.getLogger(__name__)

_EPOCH = datetime.fromtimestamp(0, timezone.utc)
_TITLE_MAX = 80

# Originators scufris tags its own sessions with. `codex exec` writes the codex
# default "codex_exec"; `codex app-server` writes the `clientInfo.name` we send
# on initialize ("scufris"). Both are ours, so the switch list must accept
# either - otherwise app_server sessions (the current default backend) vanish
# from the list even though they are on disk. See tasks/... and agent.py.
_SCUFRIS_ORIGINATORS = frozenset({"codex_exec", "scufris"})


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


class ToolCall(BaseModel):
    """One tool the agent invoked in a turn (server + tool name + status)."""

    server: str
    tool: str
    status: str


class TokenUsage(BaseModel):
    """Token counts for a request/turn, as codex reports them."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0


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
    are the real axes it does give. ``input_tokens``/``cached_input_tokens``
    describe the CURRENT context occupancy (the last request's input), so
    ``input_tokens / context_window`` is a truthful "how full" figure;
    ``output``/``reasoning``/``total`` are cumulative work over the session.
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


class TranscriptMessage(BaseModel):
    """One message in a session transcript, for re-rendering on switch."""

    role: str  # "user" | "assistant"
    text: str
    ts: datetime | None = None  # when the turn was recorded, for a UI timestamp
    # An assistant message carries the tools it ran and the turn's token usage, so
    # the UI can rebuild the "ran <tool>" chips + token count on reload (they would
    # otherwise render only on the live turn). Empty/None for user messages.
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage | None = None
    # Codex "thinking" (reasoning) that streamed live during the turn, recovered
    # from the reasoning sidecar (reasoning is NOT in the rollout - only an
    # encrypted blob). None for user turns, non-codex turns, and turns the sidecar
    # does not cover. Merged in by CodexBackend.read_transcript, not read here.
    reasoning: str | None = None


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

    Used as a read-only "last activity" signal for an agent's status
    (backends.py) - the rollout is appended to as the turn progresses.
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


def _tool_call_from_end(payload: dict[str, Any]) -> ToolCall | None:
    """A ``ToolCall`` from an ``mcp_tool_call_end`` payload, or None if malformed.

    codex records the result as a Rust-style enum (``{"Ok": ...}`` / ``{"Err":
    ...}``), so success is "Ok present".
    """
    inv = payload.get("invocation")
    if not isinstance(inv, dict):
        return None
    server = inv.get("server")
    tool = inv.get("tool")
    if not isinstance(server, str) or not isinstance(tool, str):
        return None
    result = payload.get("result")
    ok = isinstance(result, dict) and "Ok" in result
    return ToolCall(server=server, tool=tool, status="completed" if ok else "error")


def _last_usage(payload: dict[str, Any]) -> TokenUsage | None:
    """The per-request usage from a ``token_count`` payload's ``last_token_usage``."""
    info = payload.get("info")
    if not isinstance(info, dict):
        return None
    last = info.get("last_token_usage")
    if not isinstance(last, dict):
        return None
    return TokenUsage(
        input_tokens=_int(last.get("input_tokens")),
        cached_input_tokens=_int(last.get("cached_input_tokens")),
        output_tokens=_int(last.get("output_tokens")),
        reasoning_output_tokens=_int(last.get("reasoning_output_tokens")),
    )


def read_transcript(
    codex_home: Path, session_id: str | None, limit: int = 200
) -> list[TranscriptMessage]:
    """The session's conversation, so switching to it can re-render its history.

    User turns come from ``user_message``; assistant turns from the
    ``agent_message`` final answer (intermediate reasoning phases are skipped). A
    turn's ``mcp_tool_call_end`` events (which sit between the commentary and the
    final answer) are attached to that final answer as ``tool_calls``, and the
    turn's output tokens (the ``token_count`` right after the final answer) as
    ``usage`` - so the UI rebuilds the "ran <tool>" chips + token count on reload.
    Capped at ``limit`` messages (most recent kept).
    """
    if not session_id:
        return []
    path = _find_rollout(codex_home, session_id)
    if path is None:
        return []
    messages: list[TranscriptMessage] = []
    pending_tools: list[ToolCall] = []
    awaiting_usage: TranscriptMessage | None = None
    for event in _iter_events(path):
        kind = _event_kind(event)
        ts = _parse_ts(event.get("timestamp"))
        payload = _payload(event)
        if kind == "user_message":
            # A new turn - tool calls belong to the turn they ran in, not the next.
            pending_tools = []
            awaiting_usage = None
            text = payload.get("message")
            if isinstance(text, str):
                # Hide the injected steering preamble in the re-rendered history.
                text = strip_steering(text).strip()
            if isinstance(text, str) and text:
                messages.append(TranscriptMessage(role="user", text=text, ts=ts))
        elif kind == "mcp_tool_call_end":
            call = _tool_call_from_end(payload)
            if call is not None:
                pending_tools.append(call)
        elif kind == "agent_message":
            if payload.get("phase") not in (None, "final_answer"):
                continue
            text = payload.get("message")
            if isinstance(text, str) and text.strip():
                message = TranscriptMessage(
                    role="assistant",
                    text=text.strip(),
                    ts=ts,
                    tool_calls=pending_tools,
                )
                messages.append(message)
                pending_tools = []
                # Its output tokens arrive in the NEXT token_count event.
                awaiting_usage = message
        elif kind == "token_count":
            if awaiting_usage is not None and awaiting_usage.usage is None:
                awaiting_usage.usage = _last_usage(payload)
                awaiting_usage = None
    return messages[-limit:]


def reasoning_fingerprint(text: str) -> str:
    """A short, whitespace-normalized fingerprint of an assistant answer.

    Used to guard the reasoning sidecar's alignment to the transcript (see
    ``merge_reasoning``): the streamed final answer and codex's on-disk recorded
    answer can differ in trailing/collapsed whitespace, so normalize before
    hashing. This is an alignment key, not a security digest - a short prefix of
    the hash is enough.
    """
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def merge_reasoning(messages: list[TranscriptMessage], entries: list[Any]) -> None:
    """Attach sidecar reasoning to the assistant messages it covers, IN PLACE.

    ``entries`` are the sidecar's per-turn records (``answer`` fingerprint +
    ``reasoning`` text), oldest->newest - one per completed assistant turn, the
    same order assistant messages appear in ``messages``. Pair them from the tail
    (newest) backwards; accept a pair only while the answer fingerprint matches,
    and stop at the first mismatch. That way a PARTIAL or pre-existing sidecar
    (feature deployed mid-session, or a turn scufris never captured) attaches
    reasoning only to the turns it genuinely covers, and a gross mismatch yields
    no reasoning rather than a mislabeled spoiler. Pure (no I/O), so the tail
    alignment is unit-testable.
    """
    if not entries:
        return
    assistant = [m for m in messages if m.role == "assistant"]
    # strict=False on purpose: a partial/pre-existing sidecar has FEWER entries
    # than assistant messages, and tail-alignment must stop at the shorter one.
    for msg, entry in zip(reversed(assistant), reversed(entries), strict=False):
        if reasoning_fingerprint(msg.text) != entry.answer:
            break
        if entry.reasoning:
            msg.reasoning = entry.reasoning


# How many prior turns to paste as context when forking. codex-exec has no native
# "branch at turn N", so a fork re-seeds a fresh session with the earlier turns as
# text; cap it so a long history does not blow up the seed prompt.
FORK_CONTEXT_TURNS = 40


def format_fork_seed(
    context: list[TranscriptMessage],
    text: str,
    max_turns: int = FORK_CONTEXT_TURNS,
) -> str:
    """Build the seed prompt for a fork: prior turns as context + the edited text.

    Pure (no I/O) so it is unit-testable. When there is no prior context, the seed
    is just the edited text - forking the very first message is a plain new chat.
    """
    text = text.strip()
    kept = context[-max_turns:] if max_turns > 0 else context
    if not kept:
        return text
    lines = [
        "The following is an earlier conversation, provided for context.",
        "",
    ]
    for message in kept:
        who = "User" if message.role == "user" else "Assistant"
        lines.append(f"{who}: {message.text}")
    lines += [
        "",
        "End of earlier context. Continue the conversation from my next message.",
        "",
        text,
    ]
    return "\n".join(lines)


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
