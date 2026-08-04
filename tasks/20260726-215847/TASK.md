# Keep codex 'thinking' spoiler after a turn settles (ephemeral, no reload)

- PRIORITY: 5
- TAGS: feature, agents, frontend, codex, streaming
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As someone reading a codex agent's chat, I want the model's "thinking"
(reasoning) that streamed live to STILL be present as a collapsed spoiler under
the assistant turn after the turn settles (no reload), so it does not vanish the
instant the final answer lands and leave only the answer.

This is the CHEAP half of the original combined task (`20260724-141150`, now
split): frontend-only, survives settle but NOT a hard page reload. The
reload-survival half is the separate PERSISTENT follow-up (see Notes).

## Background (investigation findings, verified against current code)

- Live SSE ALREADY renders reasoning as a hidden-by-default spoiler. The event
  bus streams `reasoning_delta` events (`scufris/agent.py:88`
  `StreamReasoningDelta`, emitted from `item/reasoning/textDelta` /
  `item/reasoning/summaryTextDelta` in `_appserver_event`,
  `scufris/agent.py:351-354`). The frontend accumulates them into a
  `<details class="chat__thinking">` that starts `hidden = true`
  (`web/src/agent-chat-view.ts:481-482`, appended in `onReasoningDelta` at
  `:583-585`). So the "separate spoiler, hidden by default" UX exists live.

- It does NOT survive settle. On settle only `{role, text, reply, ts}` is pushed
  into `msgs` (`web/src/agent-chat-view.ts:547-553`, verified 2026-07-26); the
  accumulated `reasoning` string is dropped, so the next `renderChatLog` re-render
  shows the final answer only.

## Steps

- [x] Carry the streamed `reasoning` onto the settled `ChatMsg`/`ChatReply` so it
      is not dropped on the next render (`web/src/agent-chat-view.ts:547-553`).
      Add an optional `reasoning` field to the settled message shape.
- [x] Render an assistant message's `reasoning` in `renderChatLog` as a
      hidden-by-default `<details class="chat__thinking">`, reusing the exact
      styling used during the live stream.
- [x] Confirm the spoiler is collapsed (hidden/closed) by default on first render.

## Definition of Done

- After a live codex turn settles WITHOUT reload, the assistant message still
  shows a collapsed "thinking" spoiler holding the reasoning that streamed
  (manual: send a codex turn, watch it settle, confirm the thinking details is
  still present and expandable).
- The spoiler is collapsed by default (manual: confirm hidden/closed on first
  render).

## Notes

- Split out of the original combined task `20260724-141150` on 2026-07-26.
- Reload-survival is intentionally OUT of scope here; it lives in the PERSISTENT
  follow-up task (see the "Depends on" note in that task).
- CONSTRAINT (why reload is a separate, harder task): reasoning is NOT
  recoverable from disk. Codex persists reasoning as a `response_item` of
  `type: "reasoning"` whose `summary` is `[]` and whose body is an
  `encrypted_content` blob (checked `~/.codex/sessions/.../rollout-*.jsonl`).
  Plaintext thinking only exists in the live `reasoning_delta` stream, so
  `read_transcript` cannot reconstruct it - persistence needs a live-capture
  sidecar. That is the whole reason the persistent half is more than a frontend
  tweak.
- Codex-only for now; other backends not tested for reasoning streaming.
- Key files: `web/src/agent-chat-view.ts`
  (runTurn/onReasoningDelta/settle/renderChatLog), `web/src/common.ts`
  (StreamReasoningDeltaEvent and the ChatMsg/ChatReply types),
  `web/src/chat-stream.ts` (dispatchStreamEvent).
