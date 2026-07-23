# Decision: continue an in-flight per-agent turn by reattaching to the run event bus

- DATE: 20260724
- STATUS: ACCEPTED
- TASK: 20260723-001301
- TAGS: decision, agents, frontend, sse, sessions

## Context

The per-agent detail page's chat (`startAgentChat` in
`web/src/agent-chat-view.ts`) loads a settled transcript once on mount and then
only renders turns THIS browser tab POSTs to `/api/agents/<id>/chat`. It never
subscribes to the agent's live run event bus. So an orchestrator-driven turn
(`message_agent`/`run_agent`), which runs on the shared supervisor + `EventBus`
with no browser POST, does not appear live; and reloading/reselecting mid-turn
shows only the settled transcript, not the continuing turn. That is the reported
flakiness.

The backend already has the reattach seam: `GET /api/agents/<id>/events`
(`app.py` `_relay_bus_sse` + `agent_events`) relays the per-run `EventBus` as SSE
with replay-then-live semantics and a `Last-Event-ID` cursor; it 404s when the
agent has no live run. The bus is closed at turn end (`supervisor.py` `_execute`
finally -> `run.bus.close()`) but the run is retained in `_runs` up to the
history cap, so `/events` on a FINISHED run replays the last turn then closes.
The old inline run panel (`agents-view.ts`) had an `EventSource` reattach (task
20260721-112428, F0), but the F1/F2/F3 detail-page reshape dropped it.

## Decision

Restore reattach on the new per-agent chat page by reusing the existing `/events`
relay - no backend change. Specifically:

1. **Inject the reattach driver.** Add an optional `reattach` capability to
   `AgentChatConfig`, injected exactly like `streamTurn`/`loadTranscript`. jsdom
   has no `EventSource`, so the real wiring (an `EventSource` to `/events`) lives
   in `startAgentChat` and is e2e/manual-verified, while the pure component takes
   an injected function that drives the same `StreamHandlers`. This matches the
   split the codebase already uses and keeps the component unit-testable.

2. **Gate on active run status.** The real wiring opens the `EventSource` only
   when `GET /api/agents/<id>/status` reports `queued`/`running`, and CLOSES it on
   the terminal (done/error) frame. This prevents (a) replaying a finished run as
   a phantom "live" bubble and (b) the closed-bus reconnect loop (a closed bus
   yields replay-then-EOF, which a native `EventSource` would auto-reconnect into
   forever). A 404 (no active run) is handled quietly. This is F0's original
   "only open on an active run" guard, restored for the new page.

3. **Settle by pushing the bus's terminal reply, exactly like a local turn.**
   When a reattached turn ends, push the `done` frame's reply (which carries the
   text, tool_calls and usage) into the log - the same settle path a locally-POSTed
   turn uses. The reattached turn's user/prompt side comes from the mount-time
   transcript load (the backend writes the user message to the session at turn
   start, so it is already present when the page mounts mid-turn).

   An earlier revision reconciled by RE-FETCHING the transcript on settle, for
   authority over ordering/user-prompt. That was reverted: the backend persists the
   (possibly new) session id in a post-turn `on_complete` callback that runs in the
   supervisor's `finally`, AFTER the `done` frame is dispatched (`app.py`
   `_launch_agent_turn.persist` -> `mark_finished`; `supervisor.py` `_execute`). So
   an immediate reload can read `/transcript` with the session id not yet
   registered and, for a first-ever turn, get an EMPTY transcript and drop the very
   turn just streamed. Pushing the bus reply has no such race and needs no extra
   fetch; the only cost is that if the mount load raced the turn-start user-message
   write (a tiny window), the prompt line is absent until the next load.

4. **The local POST path stays the owner of locally-initiated turns.** While this
   tab is streaming its own POSTed turn (`streaming` true), the reattach driver
   does not also render a bubble; reattach is for turns started elsewhere or
   before this mount. This avoids double-rendering the same turn (which is on the
   bus AND on the POST response, since both share the run).

## Alternatives considered

- **Unify all live rendering on the events bus** - make POST `/chat` a
  fire-and-forget trigger and render every turn (local or remote) from a single
  always-open `/events` subscription. Conceptually cleaner and would make "live
  from anywhere" fall out for free. Rejected for this task: it rewrites the
  working POST-stream path, adds a trigger/attach ordering race, and needs the
  bus to stay subscribable across idle gaps; larger blast radius than the bug
  warrants. Left as a possible future consolidation.

- **Poll the transcript only (no SSE)** - re-fetch `/transcript` on an interval.
  Rejected: it does not "continue streaming" per the DoD (no token-level live
  text), and adds latency/flicker.

- **Restart-durable replay** (rebuild an in-flight turn that predates a process
  restart from the codex rollout / claude session log). Out of scope: the eventbus
  buffer is explicitly in-memory (A2+ future work), and a run does not survive a
  restart anyway.

## Consequences

Easier: the agent page shows orchestrator-driven turns live and continues an
in-flight turn across reload/reselect, reusing a tested backend relay; the
component stays jsdom-testable via the injected driver. Harder: the per-agent
chat now also polls `/status` for the reattach gate (a second small poll beside
the sidebar's existing status poll - a known desync the reshape left; unifying
them is a candidate follow-up if it bites), and each reattached settle costs an
extra transcript fetch. The unify-on-bus option remains open if live-from-anywhere
becomes a broader requirement.
