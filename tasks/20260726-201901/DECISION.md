# DECISION: Telegram live-turn rendering shape

- DATE: 20260726-201901
- STATUS: ACCEPTED
- TASK: 20260726-201901 (T6)

## Context / the fork

"See the thinking process with nice widgets for tool calls and history" is
underspecified against Telegram's hard constraints: Telegram has no real widgets
(only messages, rate-limited message edits ~1/sec, and MarkdownV2/HTML text
formatting). The plausible layouts are mutually exclusive - you cannot both
"collapse the turn into one tidy bubble" and "emit a discrete bubble per tool
call". So the concrete artifact had to be chosen before building, not inferred
(flow "stop and ask" + this repo's DECISION.md discipline).

## Options presented to the user

Layout:
- One evolving message (single bubble, edited in place, collapses on done).
- Thinking bubble + separate answer message (two bubbles).
- Message per phase (a thinking message, one message per tool call, then the
  answer - most literally "widget per event", but chattier).

Reasoning depth:
- Full streamed reasoning (throttled, tail-windowed).
- Compact status line (current step only).

## Decision (user-selected)

- Layout: **Message per phase.** A live-edited Thinking message streams the
  reasoning; each tool call is a discrete widget message; the final answer is its
  own message. Chronological order: thinking#1 -> tool A -> thinking#2 -> tool B
  -> answer (a tool call CLOSES the current reasoning bubble so the next reasoning
  opens a fresh bubble below, keeping chat order chronological).
- Reasoning depth: **Full streamed reasoning** - stream the orchestrator's whole
  reasoning into the Thinking bubble, edits throttled to a configured interval
  (first paint immediate), tail-windowed under Telegram's 4096-char cap.

## Consequences / follow-on choices (not separately asked; flagged at the plan gate)

- **Emoji + HTML widgets.** The approved previews use emoji (Thinking/tool/status
  glyphs); the natural Telegram "widget" vocabulary. This is a deliberate
  exception to the repo's ASCII-only convention (AGENTS.md; the T5
  `render_reply` is ASCII-only), scoped to the Telegram RENDERED surface only.
  Code, comments, commits, and docs stay ASCII. The final answer message stays
  plain ASCII text (render_reply) because the model's free text can contain
  `<`/markdown that HTML parse_mode would reject. Surfaced at the plan gate with
  an ASCII-only fallback offered.
- **Tool widget content = name + status only** (check/cross). `StreamTool` /
  `ToolCall` carry no per-call result payload, and adding one touches the shared
  StreamEvent used by the web SSE too - out of scope for v1. Result-detail in the
  widget is a possible follow-up, noted honestly rather than faked.
- **`telegram_stream` toggle (default True).** A False setting falls back to the
  T5 single-final-answer behaviour - the safety valve the spike wanted for the
  post-hands-on UX call. One gate inside `_render_turn`, not a second code path.
