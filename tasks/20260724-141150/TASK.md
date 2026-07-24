# Persist codex 'thinking' (reasoning) as a collapsed spoiler under each assistant turn

- STATUS: OPEN
- PRIORITY: 0
- TAGS: feature,agents,frontend,codex,streaming

## Story

As someone reading a codex agent's chat, I want the model's "thinking"
(reasoning) to persist as a collapsed spoiler under each assistant turn - so
that after the live stream settles or I reload the page, I can still expand the
reasoning that was shown live, instead of it vanishing and leaving only the
final answer.

Priority intentionally low (0) - parked for offline prioritization. Captured
now while the investigation context is fresh.

## Background (investigation findings)

Two pipelines feed a codex agent chat and they disagree on reasoning:

- Live SSE ALREADY renders reasoning as a hidden-by-default spoiler. The event
  bus streams `reasoning_delta` events (`scufris/agent.py:88` `StreamReasoningDelta`,
  emitted from `item/reasoning/textDelta` / `item/reasoning/summaryTextDelta` in
  `_appserver_event`, `scufris/agent.py:351-354`). The frontend accumulates them
  into a `<details class="chat__thinking">` that starts `hidden = true`
  (`web/src/agent-chat-view.ts:469-472`, appended in `onReasoningDelta` at
  `:569-574`). So the "separate message with a spoiler, hidden by default" UX
  already exists during the live turn.

- It does NOT survive settle or reload. On `settle` only
  `{role, text: reply.text}` is pushed into `msgs` (`web/src/agent-chat-view.ts:535-541`);
  the accumulated `reasoning` string is dropped. And `read_transcript`
  (`scufris/sessions.py:450-508`) keeps only `user_message` + the `agent_message`
  with `phase == final_answer`, explicitly skipping reasoning phases (`:490`).
  So a refresh renders the final answer only.

- CONSTRAINT verified against a real rollout: reasoning is NOT recoverable from
  disk. Codex persists reasoning as a `response_item` of `type: "reasoning"`
  whose `summary` is `[]` and whose body is an `encrypted_content` blob (checked
  `~/.codex/sessions/.../rollout-*.jsonl`). The plaintext thinking only exists in
  the live `reasoning_delta` stream. So `read_transcript` cannot reconstruct it;
  if we want persistence, Scufris must capture the live stream and store it in a
  sidecar itself.

## Decision to make offline (before building)

Do we want reasoning to persist across a hard reload, or only for the live /
just-finished turn? This splits the work into a cheap half and an involved half:

- EPHEMERAL (cheap): keep the streamed `reasoning` on the settled `ChatMsg` /
  `ChatReply` so it is not discarded on the next in-session render. Nothing new
  on the backend. Survives settle but NOT a page reload.
- PERSISTENT (involved): additionally have the backend persist reasoning per
  (session, turn) in a sidecar as it streams `reasoning_delta`, extend
  `TranscriptMessage` with an optional `reasoning` field, merge the sidecar in
  `/transcript`, and render it in `renderChatLog` as the same hidden `<details>`
  spoiler used live.

## Steps (provisional - refine after the ephemeral-vs-persistent decision)

- [ ] Decide ephemeral vs persistent (see Decision section); pick scope.
- [ ] EPHEMERAL: carry the streamed `reasoning` onto the settled `ChatMsg`/`ChatReply`
      so it is not dropped on the next render (`web/src/agent-chat-view.ts:535-541`).
- [ ] EPHEMERAL: render an assistant message's reasoning in `renderChatLog` as a
      hidden-by-default `<details class="chat__thinking">`, reusing the live styling.
- [ ] PERSISTENT (only if chosen): backend sidecar capturing `reasoning_delta`
      per (session, turn); extend `TranscriptMessage` with optional `reasoning`;
      merge into the `/transcript` response; recover on reload.

## Definition of Done

- After a live codex turn settles (no reload), the assistant message still shows
  a collapsed "thinking" spoiler with the reasoning that streamed (manual:
  send a codex turn, watch it settle, confirm the thinking details is still
  present and expandable).
- If persistent scope is chosen: after a full page reload, the same spoiler is
  present under past assistant turns (manual: reload the page, expand thinking).
- The spoiler is collapsed by default in every case (manual: confirm `hidden`/
  closed on first render).

## Notes

- This is the deferred half of the Q1/Q2 investigation. Q1 (show the
  orchestrator prompt during live reattach) is being fixed separately via /flow;
  see recent orchestrator-routing work (commits around 5ffd6b8).
- Codex-only for now; other backends not tested for reasoning streaming.
- Key files: `scufris/agent.py` (StreamReasoningDelta, _appserver_event),
  `scufris/sessions.py` (read_transcript, TranscriptMessage),
  `web/src/agent-chat-view.ts` (runTurn/onReasoningDelta/settle, renderChatLog),
  `web/src/chat-stream.ts` (dispatchStreamEvent), `web/src/common.ts`
  (StreamReasoningDeltaEvent, TranscriptMessage types).
