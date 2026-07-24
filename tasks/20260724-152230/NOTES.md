# Notes: orchestrator landing reflects the in-flight session on refresh

## Verify-first findings

- The orchestrator run IS reachable via the per-agent endpoints:
  `_require_agent(agent_id)` returns `agents.get(agent_id)`, which resolves the
  reserved `ORCHESTRATOR_ID`; `_launch_agent_turn` registers the orchestrator run
  in `agent_runs[ORCHESTRATOR_ID]`, so `/api/agents/orchestrator/status`
  (`agent_run_status`) and `/api/agents/orchestrator/events` (`agent_events`,
  relays the run bus) both work. No new endpoints needed - the landing reattach
  mirrors `startAgentChat` parameterized by these paths.

## What changed (frontend only)

- `web/src/agent-view.ts`:
  - `loadCurrentTranscript()` - the mount `loadTranscript`: fetch
    `/api/agent/sessions`, set `currentSessionId`, and if there is a `current`
    session, load its transcript (`/api/agent/session/{id}`) into the chat; empty
    current -> `[]` (welcome). Replaces the old `() => Promise.resolve([])`.
  - `reattachOrchestrator(handlers)` - the `reattach`: gate on
    `/api/agents/orchestrator/status` being running/queued, inject the driving
    prompt (`status.prompt`, Q1-A) via `onUserPrompt`, then `subscribeEvents` on
    `/api/agents/orchestrator/events`. 404/idle -> no-op.
  - `toChatMsgs` - shared transcript->ChatMsg mapper, now reused by `switchSession`.
- Tests (`web/src/agent-view.test.ts`): a local `FakeEventSource` + `stubLandingFetch`;
  auto-open-idle (transcript loads, no phantom bubble) and live-reattach (prompt +
  stream, no settle re-fetch: `/api/agent/session/s1` fetched exactly once).

## Deferred (follow-up)

- Live `onSessionStarted` pin on the landing: `createAgentChat`'s `runTurn` does
  not forward the `onSessionStarted` handler (task 1 added it to `dispatchStreamEvent`
  + `StreamHandlers`, but the shared component does not expose a per-turn hook for
  it). `onAfterTurn -> refreshSidebar` re-pins `currentSessionId` after a turn
  settles, so the only gap is forking DURING a fresh turn before it settles - a
  pre-existing edge case, out of this task's DoD. If wanted, thread an
  `onSessionStarted` config callback through `createAgentChat.runTurn`.
