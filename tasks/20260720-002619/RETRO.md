# Retro: agent backend via codex app-server

- DATE: 20260720
- VERDICT: APPROVE (1 review round)

## What went well

- Probe-first was the whole game. Before writing any production code, a throwaway
  Python script drove the real `codex app-server` JSON-RPC and printed 23 token
  deltas assembling to "1, 2, ... 8." - proving the protocol, the chatgpt auth,
  and the read-only sandbox in one run. That turned a scary experimental
  rearchitecture into a known quantity; the production runner was then a
  transcription of what the probe already did.
- Reading the generated TS bindings (`codex app-server generate-ts`) gave the
  exact method names (`initialize`/`thread/start`/`thread/resume`/`turn/start`) and
  event shapes (`item/agentMessage/delta {delta}`, `item/reasoning/textDelta`,
  `thread/tokenUsage/updated`) without guessing - the schema IS the spec.
- Additive + config-gated kept it safe: `agent_backend` defaults to `exec`, the
  Agent seam absorbed the new runner, and the SSE endpoint forwarded the new
  event kinds with no change. The end-to-end live smoke (real app_server turn ->
  14 token deltas over `/api/chat/stream`) proved the whole pipe, not just units.
- The fake-codex JSON-RPC test speaks the real handshake, so it exercises
  `_stream_app_server`'s initialize/thread/turn sequencing - not just the mapper.

## What went wrong / friction

- The first probe killed the process on the turn/start RESPONSE (which returns
  immediately) instead of reading the notification stream that follows - so it
  saw zero deltas. Fixed by reading until the `turn/completed` NOTIFICATION. This
  request-returns-then-notifications-stream shape is the key mental model for the
  protocol.
- Tool-event fidelity is best-effort: the simple probe turns did not force a tool
  call, so the `item/completed`-tool field names are inferred defensively. Flagged
  to eyeball on a real tool turn when the UI lands.

## Lessons

- `codex-app-server-for-token-streaming` (agent): `codex exec --json` is
  turn-level (no deltas, proven); token-by-token + reasoning come only from the
  experimental `codex app-server` JSON-RPC protocol on stdio. Drive it:
  `initialize` -> `thread/start` (or `thread/resume {threadId}`) -> `turn/start
  {threadId, input:[{type:text,text,text_elements:[]}]}`; the request returns
  immediately and the stream arrives as NOTIFICATIONS (`item/agentMessage/delta`,
  `item/reasoning/textDelta`, `item/completed`, `thread/tokenUsage/updated`,
  `turn/completed`). Get the method/event shapes from
  `codex app-server generate-ts`. PROBE the handshake before building. Gate behind
  a flag (experimental protocol). 20260720-002619.

## Follow-ups

- Next: the chat UI (tatr 20260720-002621) - render text deltas token-by-token, a
  "thinking" section for reasoning deltas, and a live event feed; handle the new
  SSE kinds (the current consumer treats them as unknown).
- Verify tool-event mapping on a real tool-calling app_server turn.
- Optional: a persistent app-server process (vs per-turn spawn) if startup latency
  bites; `thread/resume` already gives multi-turn continuity.
