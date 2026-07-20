# Persist tool-call chips (and per-turn usage) across session reload

- STATUS: OPEN
- PRIORITY: 50
- TAGS: bug,agent,ui

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
