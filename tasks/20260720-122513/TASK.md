# Persist tool-call chips (and per-turn usage) across session reload

- PRIORITY: 50
- TAGS: bug, agent, ui
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Tool-call chips (and the token count) render only on a LIVE turn; when you switch
to or reopen a session, the transcript re-renders WITHOUT them, so the evidence of
what the agent actually ran disappears. Restore them from the rollout, which
already records every call.

Root cause: `read_transcript` returns `TranscriptMessage{role,text,ts}` only;
`switchSession` maps to `{role,text,ts}` with no `reply`; `messageMeta` builds
chips from `reply.tool_calls` (set only on `onDone`).

## Notes

- Spike: tasks/20260720-122301/SPIKE.md.
- User feedback: "the tool chip disappears when I enter the chat again."
- The rollout records each call as an `event_msg` `mcp_tool_call_end`
  `{call_id, invocation:{server,tool}, duration, result}`, ordered between the user
  turn and the following `agent_message`. Harvest these in `read_transcript`,
  correlate to the next assistant message, and carry `tool_calls` (and ideally the
  per-turn token usage) through `TranscriptMessage` -> frontend rebuilds the reply
  meta so chips survive a reload.
- Natural extension (optional, note at /plan): clicking a chip could show
  args/result/duration - the rollout has all of it.
- Backend + frontend; escape everything; keep render side-effect-free for jsdom.

## Implementation

Reproduced first with a failing `read_transcript` test (a rollout with a
`user_message -> commentary agent_message -> mcp_tool_call_end -> final_answer
agent_message -> token_count`), which errored on the missing field, then fixed:

- `sessions.py`: `ToolCall`/`TokenUsage` MOVED here from `agent.py` (so
  `TranscriptMessage` can carry them without an import cycle - `agent.py` already
  imports from `sessions.py`); `agent.py` re-imports them (re-exported as
  `scufris.agent.ToolCall`/`.TokenUsage` for existing callers). `TranscriptMessage`
  gains `tool_calls: list[ToolCall]` + `usage: TokenUsage | None`.
- `read_transcript` now correlates: `mcp_tool_call_end` events accumulate per turn
  (reset on each `user_message`, since a turn's calls sit between its commentary and
  its final answer) and attach to that turn's `final_answer` message; the turn's
  output tokens come from the `token_count` event right after the final answer
  (buffered via `awaiting_usage`). Helpers `_tool_call_from_end` (result is a
  Rust-style `{"Ok"|"Err"}` enum -> status) and `_last_usage`.
- `common.ts`: `TranscriptMessage` gains `tool_calls` + `usage`.
- `agent-view.ts`: new exported `transcriptReply(m)` rebuilds a `ChatReply` from a
  reloaded message (undefined when no tools/usage); `switchSession` uses it so
  `messageMeta` re-renders the chips + token count on reload, not just live.

## Verification

- Reproduction test now green; backend 132 pytest, frontend 85 (+3), ruff/mypy
  clean. `transcriptReply` unit-tested + a render test that a reloaded message shows
  the "ran <tool>" chip.
- REAL-DATA check: `read_transcript` on an actual rollout returns the assistant
  message with `tool_calls=[host_stats, disk_usage, list_processes]`, out_tok=497.
