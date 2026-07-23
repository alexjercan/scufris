# Goal: harden the orchestrator<->sub-agent session view on the agent page

- DATE: 20260724
- UMBRELLA TASK: 20260724-001517
- LANDING SCOPE: squash-merge each task to local `master` (flow default). Do NOT
  push; pushing is the user's call at Finish.

## Goal

The per-agent detail page (`/agents/<id>`) renders the agent's conversation, but
it only ever loads a settled transcript once on mount and then streams turns that
THIS browser tab POSTs itself. It never subscribes to the agent's live run event
bus. So a turn driven from elsewhere - the orchestrator calling
`message_agent`/`run_agent` against a sub-agent, which runs on the shared
supervisor + event bus - does not appear live, and reloading or reselecting the
agent mid-turn shows the settled transcript but does NOT continue the in-flight
turn. That is the reported flakiness.

The backend already has everything needed: `GET /api/agents/<id>/events` relays
the per-run `EventBus` as SSE with replay-then-live semantics and a
`Last-Event-ID` reconnect cursor (tested: `test_agent_events_relay`, the eventbus
replay tests). The persisted `SessionRegistry` (dependency task 20260723-001251,
CLOSED) gives stable, backend-keyed session ids underneath. The fix is on the
frontend: wire the per-agent chat to that events relay so a reselect/reload
rebuilds the full transcript AND continues an in-flight turn's streaming.

Tool-call chips + per-turn usage already replay across reload via
`transcriptReply` (shared by both chats) - that half of the Direction is already
satisfied and is verified here, not rebuilt.

## Done means

1. On mounting the per-agent page while a turn is in flight (queued/running), the
   chat rebuilds the settled transcript AND attaches to the live run, rendering
   the in-flight turn's streaming text/tool/reasoning frames to completion.
   (test: a jsdom component test drives an injected reattach capability that
   replays an in-flight turn; asserts the live bubble renders and settles)
2. A turn started elsewhere (orchestrator-driven `message_agent`/`run_agent`)
   is what this covers: reattach uses the run event bus, not a locally-POSTed
   turn. Reattach is gated on active run status so a FINISHED run is not replayed
   as a phantom live bubble and the closed-bus EventSource does not reconnect-loop.
   (test: reattach is a no-op / does not open when status is idle/done)
3. Locally-initiated turns still render exactly as today (the POST `/chat` stream
   owns rendering; reattach does not double-render them).
   (test: existing chat-view tests stay green; a test asserts no double bubble)
4. `npm run ci` passes in `web/` (prettier + eslint + vitest + webpack build).
   (cmd: `cd web && npm run ci`)
5. The backend QA gate stays green (no backend change expected; if any is made it
   is covered). (cmd: `nix flake check` or the fast `python -m pytest` equivalent)

Overall: reselecting or reloading an agent mid-turn shows the full transcript and
continues streaming, live and on reload, without phantom or duplicated turns.

## Tasks

- [ ] 20260723-001301 (p30, web) Harden the orchestrator<->sub-agent session view
      on the agent page

## Decisions (load-bearing, architectural)

- 20260723-001301 DECISION.md: reuse the existing `/events` bus relay; add an
  INJECTED `reattach` capability to the shared chat component, gated on active run
  status, reconciling by transcript reload on settle (ACCEPTED)

## Manual acceptance (batched for the user at Finish)

- (pending) 20260723-001301: with a real backend, drive a sub-agent turn from the
  orchestrator (or a long `/chat` turn), then reload / reselect the agent page
  mid-turn and confirm the transcript rebuilds and the in-flight turn keeps
  streaming to completion, with no duplicated or phantom bubbles.
