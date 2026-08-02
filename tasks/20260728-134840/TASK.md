# Cancel in-flight chat runs: square stop button in chat UI + orchestrator cancel_agent tool + CANCELLED state

- PRIORITY: 0
- TAGS: feature, agents, frontend, backend, mcp, ui, streaming, backlog
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As a user I want to cancel an in-flight chat run. In any chat UI (the
orchestrator landing chat and every sub-agent chat), while a turn is streaming
the send/Enter button becomes a square STOP button; clicking it cancels the run.
I also want the orchestrator to be able to cancel a sub-agent's run on command
("cancel that subagent") via a tool, so cancellation works both manually
(go to the agent's chat and hit stop) and by instruction to the orchestrator.

A cancelled turn is a first-class, user-initiated stop - NOT an error. The
partial assistant output streamed so far stays in the transcript, marked as
interrupted/cancelled, so the user (and the next turn's context) keeps what was
produced before stopping.

## Understanding (flow: gathered before planning)

Run engine (already cancel-ready):
- Every turn is an `asyncio.Task` in `Supervisor` (`scufris/supervisor.py`),
  keyed by `run_id`; `app.py` maps `agent_runs[agent.id] -> run_id`.
- `_execute` already catches `asyncio.CancelledError` and records
  `run.error = "cancelled"`; the `finally` closes the EventBus (ends SSE relays)
  and releases the serialize slot. The Claude backend's `stream()` `finally`
  `proc.kill()`s its subprocess, so a real task-cancel aborts upstream, not just
  detaches the reader (must ensure the generator is `aclose`d so that finally runs).
- The orchestrator is itself an agent record (`ORCHESTRATOR_ID = "orchestrator"`)
  and lives in `agent_runs`, so ONE agent-keyed cancel path covers the landing
  chat and sub-agents alike.

What is missing = only the user-facing trigger + a distinct terminal state:
1. `Supervisor.cancel(run_id)` -> cancel the task.
2. HTTP cancel endpoint keyed by agent id (covers orchestrator via its id).
3. Orchestrator MCP tool `cancel_agent(agent_id)` in `scufris/mcp_server.py`.
4. Frontend: swap the send button for a square STOP button while streaming
   (`web/src/agent-chat-view.ts`), wired to POST the cancel endpoint AND abort
   the local fetch. Config-driven like `streamTurn` (add a `cancelTurn`/cancel
   URL per view: `/api/chat/*` for orchestrator, `/api/agents/{id}/*` per agent).
5. New `AgentState.CANCELLED` (`scufris/enums.py`) distinct from ERROR, threaded
   through the persist/outcome path (`agent_store.py`), status text
   (`mcp_server._agent_status_text`), and `pending_agents` (a user-cancelled
   agent is NOT pending to the orchestrator). Map the run's "cancelled" error to
   CANCELLED rather than ERROR.
6. Partial assistant output is kept and marked interrupted in the transcript
   (frontend `stop()`/settle path), not discarded.

Design decisions (confirmed with user):
- Terminal state: NEW `AgentState.CANCELLED`, not reuse of ERROR.
- Partial output: KEEP, marked interrupted; retained in the visible transcript
  so a follow-up turn can build on the partial answer.

## Key files

- `scufris/supervisor.py` - add `cancel(run_id)`; ensure generator aclose.
- `scufris/app.py` - cancel endpoint(s); map cancelled -> CANCELLED in persist.
- `scufris/enums.py` - `AgentState.CANCELLED`.
- `scufris/agent_store.py` - outcome/state threading, pending exclusion.
- `scufris/mcp_server.py` - `cancel_agent` tool + status text.
- `web/src/agent-chat-view.ts` - stop button + cancel wiring + partial keep.
- `web/src/chat-stream.ts` - AbortController into the fetch.
- `web/src/agent-view.ts` - orchestrator cancel URL wiring.
- Tests: `tests/` (FastAPI route + supervisor cancel), `web/src/*.test.ts`.

## Steps

Backend (cancel engine + endpoint):

- [x] Add `CANCELLED = "cancelled"` to `AgentState` in `scufris/enums.py`.
- [x] In `scufris/supervisor.py`: add `cancel(run_id) -> bool` that looks up the
      live `_Run`, sets an explicit `run.cancelled = True` flag (distinct from
      stall/budget/error), and cancels `run.task`. Surface `cancelled` on the
      run snapshot / `RunState` so the persist callback can read it. Do NOT rely
      on the `run.error == "cancelled"` string.
- [x] In `scufris/supervisor.py`: the backend stream generator is `aclose()`d
      when a run is cancelled (already done in `_drain`'s finally), so each
      backend's `finally` runs (Claude `proc.kill()`, HTTP client close) and
      generation is truly aborted, not just detached. Verified by the supervisor
      test's closed-flag finally.
- [x] In `scufris/app.py` `_launch_agent_turn` persist callback: when the
      snapshot is `cancelled`, mark the agent `AgentState.CANCELLED` with message
      "cancelled" (precedence over ERROR, mirroring the existing error path).
- [x] In `scufris/app.py`: add `POST /api/agents/{agent_id}/cancel` (async) ->
      look up `agent_runs[agent_id]`, call `supervisor.cancel(run_id)`; 200 with
      `{agent_id, cancelled: true}` on success, 404 when the agent has no active
      run. This same path serves the orchestrator via `ORCHESTRATOR_ID`.
- [x] `scufris/agent_store.py`: `mark_finished` already coerces + threads any
      `AgentState` (incl. CANCELLED); `pending_outcomes()` already excludes it
      (only WAITING/REPORTED/ERROR are pending). No change needed - verified by
      test. `wake.py` likewise only wakes on those three, so a cancel never wakes.

Orchestrator tool:

- [x] In `scufris/mcp_server.py`: add a `cancel_agent(agent_id: str)` tool that
      POSTs `/api/agents/{agent_id}/cancel` (mirroring `run_agent`); refuses the
      orchestrator (cannot cancel its own run from within). Render the `CANCELLED`
      outcome in `_agent_status_text`.

Frontend (stop button + wiring + partial keep):

- [x] In `web/src/chat-stream.ts`: thread an `AbortSignal` into `streamPost` /
      `streamChatTurn`; treat `AbortError` (fetch + reader.read) as a clean stop
      (swallowed, not routed to onError).
- [x] In `web/src/agent-chat-view.ts`: add a `cancelTurn: () => Promise<void>`
      config hook and hold an `AbortController` per in-flight turn. While
      `streaming`, render the composer submit button as a square STOP control
      (`is-stopping` class, aria-label "stop", cleared label); its activation
      aborts the local fetch AND calls `cancelTurn()`. On settle, restore "send".
- [x] In `web/src/agent-chat-view.ts`: on cancel, KEEP the partial assistant
      message tagged with a `chat__cancelled` "(cancelled)" marker (via a
      `ChatMsg.cancelled` flag), not discarded.
- [x] Wire `cancelTurn` per view: orchestrator landing (`web/src/agent-view.ts`)
      -> `POST /api/agents/orchestrator/cancel`; per-agent -> `POST
      /api/agents/{id}/cancel`.
- [x] Add square-button styling for `.chat__send.is-stopping` (+ the
      `.chat__cancelled` tag, + `.agents__badge--cancelled`) in `web/src/style.css`.

Tests + docs:

- [x] Backend tests: supervisor cancel marks CANCELLED and acloses the generator;
      the cancel route stops a live run (200) + 404 idle/unknown; the orchestrator
      run cancels via its id; a cancelled agent is not pending; the `cancel_agent`
      MCP tool hits the endpoint (+ refuses orchestrator, reports no active run).
- [x] Frontend tests in `web/src/agent-chat-view.test.ts`: the button becomes a
      stop control while streaming; activating it aborts + calls the cancel hook
      and restores to send; partial output is kept and marked; no stop affordance
      when `cancelTurn` is unwired.
- [x] Update `CHANGELOG.md` (Added) and the README Agents section.

## Definition of Done

- `Supervisor.cancel(run_id)` cancels an in-flight run, marks the snapshot
  cancelled, acloses the stream generator (backend `finally` runs), and reaches
  a CANCELLED outcome - not ERROR
  (test: `test_cancel_marks_cancelled_and_closes_stream`).
- `POST /api/agents/{id}/cancel` stops a live run and returns 200
  `{cancelled: true}`, persisting the agent as `CANCELLED`, and a cancelled agent
  is NOT in `pending_outcomes`; it returns 404 when the agent has no active run or
  is unknown
  (test: `test_cancel_endpoint_stops_run_and_marks_cancelled`,
  test: `test_cancel_endpoint_404_when_idle_or_unknown`).
- The orchestrator landing chat is cancellable through the same path via
  `ORCHESTRATOR_ID` (test: `test_cancel_orchestrator_run`).
- The orchestrator MCP tool `cancel_agent(agent_id)` cancels a sub-agent's run
  and refuses the orchestrator itself
  (test: `test_cancel_agent_posts_cancel`, test: `test_cancel_agent_refuses_orchestrator`).
- While a turn streams, the composer button renders as a square stop control
  (not "send"); activating it aborts the request, calls the cancel hook, and
  restores to "send" when the turn settles
  (test: `` `agent-chat-view` vitest: "stop button cancels a streaming run" ``).
- Partial assistant output streamed before a cancel stays in the transcript,
  marked interrupted
  (test: `` `agent-chat-view` vitest: "partial output kept and marked on cancel" ``).
- CHANGELOG records the feature
  (cmd: `grep -rn "cancel" CHANGELOG.md`).
- Full gate is green: ruff + mypy + pytest and the web unit tests
  (cmd: `python -m pytest`), (cmd: `cd web && npm test`).
- manual: in the running app, sending a message shows a square stop button;
  clicking it stops streaming and the partial reply stays marked cancelled;
  telling the orchestrator "cancel agent X" (or opening X's chat and hitting
  stop) stops that sub-agent's run.

## Notes

- Decision recorded in `tasks/20260728-134840/DECISION.md`: user-cancel is a
  distinct `AgentState.CANCELLED`, not reuse of ERROR.
- One agent-keyed cancel path (`/api/agents/{id}/cancel`) covers the orchestrator
  because `ORCHESTRATOR_ID = "orchestrator"` lives in `agent_runs` like any agent.
- Cancellation truth: the supervisor must `aclose()` the stream generator so the
  Claude backend's `finally` (`proc.kill()` in `scufris/backends.py`) actually
  terminates the subprocess; otherwise cancel only detaches the reader.
- Do not key CANCELLED off the `run.error == "cancelled"` string; add an explicit
  `run.cancelled` flag on the run/snapshot so a real backend error that happens
  to say "cancelled" is not misclassified.
- New state route audit: grep everything that branches on `AgentState` (status
  text, pending logic, UI badges, telegram) and confirm CANCELLED renders
  sensibly everywhere it can now appear.
- "Retained for the next turn" is best-effort at the UI/transcript layer; the
  backend session (codex/claude/opencode) may not persist a mid-turn partial.
  Keep + mark it in the visible transcript; do not block on backend session
  persistence of the partial.

## Close-out

Implemented as planned; design/impl record in `NOTES.md`, decision in
`DECISION.md`. Notable adaptations from the plan:

- `agent_store.py` needed NO change: `mark_finished` already coerces + threads any
  `AgentState`, and `pending_outcomes`/`wake.py` only act on WAITING/REPORTED/ERROR
  - so CANCELLED is correctly non-pending and never wakes the orchestrator, for
  free.
- Cancel endpoint returns 200 `{cancelled: true}` (mirroring the other agent-run
  responses), not the originally-planned 204. DoD updated to match.
- The supervisor's `_drain` already `aclose()`d the stream generator, so the
  "ensure aclose" step was verify-only (confirmed by the supervisor test's
  finally-ran flag).

Gate: `ruff`, `mypy`, `python -m pytest` (561 passed), web `prettier`/`webpack`
build + `vitest` (189 passed) all green. One `manual:` DoD item (drive the running
app) is left for the reviewer/user to accept.

## Lessons applied (from LESSONS.md)

- `supervisor-endpoints-must-be-async`: the cancel endpoint calls
  `supervisor.cancel` (touches the loop) -> it MUST be `async def`.
- `concurrent-request-test-needs-async-httpx-not-testclient-stream`: to cancel
  an IN-FLIGHT run in a test, hold it open with `httpx.AsyncClient(ASGITransport)`
  on one loop - `TestClient.stream` buffers and deadlocks.
- `assert-terminal-outcome-on-the-durable-record-not-status`: assert the cancel
  outcome on the durable record (`GET /api/agents/{id}` / OutcomeStore / pending),
  not `/api/agents/{id}/status` (which reports live RunPhase).
- `tool-reachable-by-two-runners-needs-a-test-per-runner`: exercise the actual
  `cancel_agent` MCP tool AND the HTTP endpoint it calls.
- `strenum-field-needs-coercion-on-unvalidated-writes` /
  `strenum-fields-take-the-member-not-the-raw-str-in-typed-callers`: pass
  `AgentState.CANCELLED` as the member; coerce on `model_copy`/attr-assign writes.
- `assert-form-control-value-not-textcontent`: assert the stop button by its
  state (class/aria-label/property), not vacuous textContent.
