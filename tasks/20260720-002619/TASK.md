# Agent backend via codex app-server: stream token/reasoning/tool deltas

- STATUS: OPEN
- PRIORITY: 40
- TAGS: feature, agent, spike

## Goal

`codex exec` cannot stream token-by-token (proven in the spike). Add a second
agent backend that drives the experimental `codex app-server` (a persistent
JSON-RPC-over-stdio server) and consumes its notification stream, so a turn yields
live events: `outputDelta` (assistant text token-by-token),
`ReasoningTextDelta`/`ReasoningSummaryTextDelta` ("thinking"), thread-item / tool /
process events, and final usage. Wire it behind the existing `Agent` seam,
config-gated (`agent_backend = exec | app_server`) so `codex exec` stays the
fallback + CLI path. Forward the new events over the existing `/api/chat/stream`
SSE endpoint as new event kinds (text-delta, reasoning-delta, event, done).

## Notes

- Spike: tasks/20260720-002611/SPIKE.md.
- PLAN PROBE-FIRST: the opening step is a throwaway-grade prototype that completes
  the app-server JSON-RPC handshake (initialize + start a turn) and captures a
  real streamed turn's `outputDelta`/reasoning deltas, to confirm the protocol +
  chatgpt auth + sandbox wiring BEFORE building the production client. Use
  `codex app-server generate-json-schema` / `generate-ts` for the method + event
  shapes.
- EXPERIMENTAL protocol: pin behind the config flag, keep `codex exec` as
  fallback, and re-inspect the schema on codex upgrades.
- Depends on nothing hard; blocks tatr 20260720-002621 (the UI).
- Keep the injectable-runner/seam patterns and the SSE streaming shape from tatr
  20260719-223103; add event kinds rather than replacing them.
