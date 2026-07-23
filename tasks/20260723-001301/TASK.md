# Harden the orchestrator<->sub-agent session view on the agent page

- STATUS: OPEN
- PRIORITY: 30
- TAGS: bug,agents,frontend,ui

## Story

As the operator, I want the orchestrator<->sub-agent conversation to render reliably
on the agent's page (live, and on reload / reselect), because today it works but is
flaky.

## Context

Orchestrator turns against a sub-agent (`message_agent` / `run_agent`) already run on
the shared supervisor + event bus and land on the agent page. "Sometimes not well"
is reattach/replay robustness - the same class as several open UI items, not a new
mechanism.

Root cause (found in planning): the per-agent chat entry (`startAgentChat` in
`web/src/agent-chat-view.ts`) only calls `loadTranscript()` once on mount and then
streams turns THIS tab POSTs to `/api/agents/<id>/chat`. It never subscribes to the
agent's live run event bus. The backend already relays that bus at
`GET /api/agents/<id>/events` (replay-then-live, `Last-Event-ID` cursor; tested by
`test_agent_events_relay` + the eventbus replay tests). The old inline run panel
(`agents-view.ts`) had an `EventSource` reattach from task 20260721-112428 (F0), but
the F1/F2/F3 detail-page reshape dropped it and never re-added it to the new chat
page. The persisted `SessionRegistry` (dep 20260723-001251, CLOSED) already gives
stable, backend-keyed session ids underneath.

Tool-call chips + per-turn usage already replay across reload via `transcriptReply`
(shared by both chats). That half of the Direction is DONE; this task verifies it,
it does not rebuild it.

## Direction

- [ ] Harden SSE reattach-on-select and transcript replay for the per-agent session
      view: a reselect/reload must rebuild the full transcript AND continue an
      in-flight turn.
- [x] Persist/replay tool-call chips + per-turn usage across reload for agent
      sessions (the orchestrator chat already does this - reuse it). Already shared
      via `transcriptReply`; verify with a test rather than rebuild.
- [x] Do the persisted session registry task FIRST - dep 20260723-001251 CLOSED.

## Steps

- [ ] Reproduce first (bug discipline): add a failing jsdom component test in
      `web/src/agent-chat-view.test.ts` that mounts the per-agent chat with an
      injected `reattach` capability simulating an in-flight turn (a settled
      transcript PLUS live text/tool frames that then settle), and asserts the live
      bubble renders and settles into the transcript. Watch it fail (no reattach
      capability yet).
- [ ] Add an optional `reattach` capability to `AgentChatConfig` in
      `agent-chat-view.ts`, injected like `streamTurn`/`loadTranscript` so jsdom
      tests drive it without a real `EventSource`. Shape: a function that, given the
      same `StreamHandlers` the local turn uses, attaches to the live run and drives
      a pending bubble, returning a disposer. Reuse the existing `runTurn` internals
      for the live bubble so live and reattached turns render identically.
- [ ] On mount, AFTER `loadTranscript()` resolves, invoke `reattach` when present.
      While a local POST turn is streaming (`streaming` true), do NOT drive a
      reattach bubble - the POST stream owns rendering. On the reattached turn's
      terminal frame (done/error), reconcile by reloading the transcript
      (authoritative) rather than appending, so no turn is duplicated.
- [ ] Wire the REAL reattach in `startAgentChat`: poll `GET /api/agents/<id>/status`;
      when the state is `queued`/`running` and no stream is open, open an
      `EventSource` to `/api/agents/<id>/events`, route its frames through
      `parseSseFrames`-equivalent handling (the bus emits the same `StreamEvent`
      frames), and CLOSE it on the terminal frame so the closed bus never triggers an
      `EventSource` auto-reconnect loop. Gate on active status so a FINISHED run is
      never replayed as a phantom live bubble. Handle the 404 (no active run) quietly.
- [ ] Verify the already-shared chip/usage replay: a jsdom test that a reloaded
      per-agent transcript with `tool_calls` + `usage` renders the "ran <tool> N tok"
      meta line (via `transcriptReply` -> `messageMeta`). Assert a DISTINCT value,
      not a default.
- [ ] Regression guard: a test that a locally-initiated turn (POST stream) with
      reattach ALSO configured does not render a duplicate bubble.
- [ ] `cd web && npm run ci` green (prettier + eslint + vitest + webpack build).
      Run `prettier --write` before the gate. Backend: no change expected; if any is
      made, `python -m pytest` (from the worktree) covers it and stays green.
- [ ] Docs sync: check `README` / any agent-page docs against the diff; update in
      THIS task if the reattach behavior is described anywhere.

## Definition of Done

- Reselecting or reloading an agent mid-turn shows the full transcript and continues
  streaming.
  (test: jsdom component test drives injected reattach - in-flight turn renders live
  and settles; manual: real orchestrator-driven sub-agent turn, reload mid-turn)
- Reattach does NOT open / render for an idle or finished run (no phantom bubble, no
  EventSource reconnect loop).
  (test: reattach is a no-op when status is idle/done)
- A locally-POSTed turn is not double-rendered when reattach is configured.
  (test: no duplicate bubble)
- Tool-call chips + per-turn usage still replay across reload for agent sessions.
  (test: reloaded transcript renders the "ran <tool> N tok" meta with a distinct value)
- `cd web && npm run ci` passes. (cmd: `cd web && npm run ci`)
- Backend QA gate green. (cmd: `python -m pytest` in the worktree, or `nix flake check`)

## Notes

- Overlaps existing backlog (all CLOSED, coordinated not duplicated):
  `20260721-112428` (F0 SSE reattach on select - the mechanism this restores for the
  new page), `20260720-020356` (stream tokens end-to-end), `20260720-122513` (persist
  tool-call chips across reload - the `transcriptReply` this reuses).
- Depends on the `(agent_id -> session_id)` persisted registry task 20260723-001251
  (CLOSED).
- Design recorded in DECISION.md (this folder): reuse `/events`, inject reattach,
  gate on active status, reconcile by transcript reload.
- jsdom has no `EventSource`; keep the real `EventSource` wiring in `startAgentChat`
  (e2e/manual-verified) and inject the driver so the pure component stays testable -
  same split the codebase already uses for `streamTurn`/`loadTranscript`.
