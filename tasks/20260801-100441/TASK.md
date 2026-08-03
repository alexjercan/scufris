# Extract the orchestrator-turn and agent-run services

- PRIORITY: 71
- TAGS: refactor, v0.2.0, agents, backend, telegram
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100425

## Story

As a maintainer, I want one transport-independent orchestrator-turn service and
one agent-run service, so that the landing chat, Telegram, and the wake bridge
stop each implementing their own version of starting and finishing a turn.

## Steps

- [ ] Add the failing tests first, in a new `tests/test_orchestrator_service.py`:
      `test_orchestrator_transports_share_turn_service` (the landing chat, the
      Telegram `on_message` callback and the wake bridge's launch all reach
      `AgentRunService.launch` - one recording fake service, three callers),
      `test_agent_run_lifecycle_is_owned_by_the_run_service` (launch, the 409
      guard, cancel, status, events bus, request_input, report_back,
      acknowledge and fork all go through the service, driven with no FastAPI
      app), `test_chat_stream_events_are_unchanged` (the `/api/chat/stream` SSE
      frames and `id:` sequence for one scripted turn match the pre-extraction
      capture, image-attachment error frame included), and
      `test_the_services_carry_no_transport_imports` (`scufris/orchestrator/`
      sources contain no `fastapi` or `telegram` import).
- [ ] Add `scufris/orchestrator/errors.py`: the typed refusals the routes
      currently express as `HTTPException` - `RunAlreadyActive` (409),
      `NoActiveRun` (404), `AgentDisabled` (503), `AgentProjectMissing` (422),
      `TurnFailed` (503, a terminal `StreamError`), `TurnEndedWithoutReply`
      (500). Map them to statuses in ONE place, `scufris/api/errors.py`, beside
      the existing `hostd_http_error` table.
- [ ] Add `scufris/orchestrator/runs.py`: `AgentRunService`, owning the
      supervisor handle, the `agent_runs` registry, the `launching_runs` claim
      set, `_agent_run_active`, the whole of `_launch_agent_turn` (backend
      resolve, claim, `turn_stream`, the reasoning sidecar write, the `persist`
      completion callback), `drain` (was `_drain_turn`), `cancel`, `status`,
      `bus`, `request_input`, `report_back`, `acknowledge`, `fork_seed`, and
      `aclose`. Agent and project resolution (`_require_agent*`) moves here and
      raises `AgentNotFound` / `AgentProjectMissing`, not `HTTPException`.
- [ ] Give `AgentRunService` an `on_complete(hook)` registration in the shape
      `HostApprovalService.on_proposed` already uses, called in registration
      order at the end of `persist`. `create_app` registers the wake bridge's
      `on_run_complete` and then `_drain_deferred_decision`, preserving today's
      order; the WakeBridge's `launch`/`is_orchestrator_busy` become
      `runs.launch`/`runs.active` and stop closing over `create_app`.
- [ ] Add `scufris/orchestrator/turn.py`: `OrchestratorTurnService` over the run
      service - `send(message) -> AgentReply` (`/api/chat`), `stream(message,
      image_paths=None, on_done=None) -> (run_id, bus)` (`/api/chat/stream`,
      Telegram, the wake bridge), `reset()` (`/api/chat/reset`, the Telegram
      `/new`), `cancel()` (`/cancel`), and `busy()`. It owns the
      `settings.agent_enabled` check and the `ORCHESTRATOR_ID` lookup that the
      three transports each repeat today.
- [ ] Move `build_telegram_callbacks` out of `app.py` into
      `scufris/telegram/orchestrator.py`, rewritten over
      `OrchestratorTurnService`: it catches `RunAlreadyActive` instead of
      `HTTPException.status_code == 409`, and its user-facing strings join
      `telegram/text.py`. Re-export it from `scufris.app` is NOT kept - update
      `tests/test_telegram_app.py`'s import.
- [ ] Reduce the route bodies to translation: `/api/chat`, `/api/chat/stream`,
      `/api/chat/reset` and the nine `/api/agents/{id}/...` lifecycle handlers
      call a service method and map a typed refusal onto a status. `/chat`'s
      live-approval 409 and `/fork`'s orchestrator 409 stay in the route (they
      are auth/identity translation, not run lifecycle). `_write_image_to_temp`
      and its SSE error frame stay in the route (DECISION.md 3).
- [ ] Wire it in `create_app`: build `AgentRunService` where `supervisor` is
      built today, publish it as `app.state.runs` alongside the existing
      `app.state.supervisor` (kept - `tests/test_app.py` and the route-contract
      state-key test read it), and have the lifespan call `runs.aclose()` in
      place of `supervisor.aclose()`.
- [ ] Update `scufris/README.md`: the module map gains
      `orchestrator/`, section 7's "a route translates, it does not decide" rule
      names the new services, and section 4/8 stop pointing at
      `_launch_agent_turn` by name. `scufris/wake.py`'s module docstring says
      the same.
- [ ] Run the checks: `ruff check . && ruff format --check . && mypy . &&
      python -m pytest`, then `python scripts/check_file_size.py` (each new
      module under the 600-line `SOURCE_CAP`; `scufris/app.py` stays
      allowlisted - the successor task removes the entry), then `tatr check`.

## Definition of Done

- All three transports launch through the same service
  (test: `test_orchestrator_transports_share_turn_service`).
- Run lifecycle is owned by one service for every caller
  (test: `test_agent_run_lifecycle_is_owned_by_the_run_service`).
- The services carry no transport imports
  (cmd: `test -d scufris/orchestrator && ! rg -n "fastapi|telegram"
  scufris/orchestrator/`; red on base because the directory does not exist).
- Streaming and SSE behavior is unchanged
  (test: `test_chat_stream_events_are_unchanged`).
- The public route surface, `app.state` keys and lifespan ownership are
  untouched by the move
  (cmd: `python -m pytest tests/test_route_contract.py`).
- Every new module is inside the source cap
  (cmd: `python scripts/check_file_size.py`).
- Existing API and browser suites pass without drift
  (cmd: `python -m pytest && cd web && npm run ci && npm run test:e2e`).

## Notes

- Epic: 20260729-102145.
- Depends on the route characterization task; its route-contract test is the
  gate for this one too.
- Do not invent the future conversation schema here. The required seam is one
  transport-independent orchestrator service that 20260729-220835 can place a
  durable conversation around later.
- Move models and helpers only when their ownership becomes clearer. A
  mechanical one-file-to-many split with the same coupling is not the goal.

### Discovered while planning

- The Steps as written named `scufris/telegram/turn.py` as a turn path to
  extract from. It is not one: that module is pure RENDERING (`drive_turn` /
  `_render_turn` lay `StreamEvent`s out over Telegram messages) and stays
  exactly as it is. The Telegram turn LAUNCH path is
  `app.py::build_telegram_callbacks`, which is what moves.
- The DoD's original `! rg -n "fastapi|telegram" scufris/services/` proof is
  green on base for the wrong reason: `rg` on a missing directory exits
  non-zero, which `!` turns into a pass. Replaced with a `test -d &&` guard,
  which is red on base and red again if the directory is ever emptied.
- `_launch_agent_turn` (`app.py:1557-1735`) is the single seam every caller
  already funnels through - `/api/agents/{id}/run|chat|fork`, `/api/chat`,
  `/api/chat/stream`, `build_telegram_callbacks`, `_wake_launch` and
  `_deliver_decision`. The extraction is therefore a move of one function plus
  its state (`agent_runs`, `launching_runs`, `supervisor`), not a rewrite.
- Three closures currently catch `HTTPException` to mean "409, already active":
  `_deliver_decision` (app.py:1798), `_wake_launch` (app.py:1880) and the
  Telegram `on_message` (app.py:641). All three become `except
  RunAlreadyActive`, which is strictly narrower - today a 404 or 503 from the
  launch path is silently swallowed as "busy" by two of them.
- `persist` already runs `wake_bridge.on_run_complete` then
  `_drain_deferred_decision`, both AFTER `mark_finished` and past the serialize
  key release. That ordering is load-bearing (lesson
  `serialize-then-launch-self-deadlocks-on-shared-key`) and is what the
  `on_complete` hook list must preserve - assert the order in the new test.
- `SOURCE_CAP` is 600 lines (`scripts/check_file_size.py`). `_launch_agent_turn`
  plus the lifecycle handlers is ~450 lines with docstrings, so `runs.py` is
  close to the cap on its own - hence turn/runs/errors as three modules rather
  than one, which is also what the task asks for.
- `app.py` is 2923 lines today and is expected to land near ~2500. It stays on
  the file-size allowlist; 20260729-103712 is the task that removes the entry.

### Assumptions

- `app.state.supervisor` stays published. The route-contract test asserts the
  exact `app.state` key set and `tests/test_app.py:591` calls
  `list_runs()` on it, so removing it is a contract change this task did not
  ask for. `app.state.runs` is added beside it, and the route-contract
  expectation gains that one key.
- `AgentReply`, `RunStarted`, `AgentRunStatus` and the other pydantic response
  models stay in `app.py`. They are the HTTP surface; moving them is the
  successor task's problem, and the Notes above say not to move models whose
  ownership is not yet clearer.
- The browser suite (`npm run test:e2e`) is expected to pass unchanged. If it
  needs a running server that this environment cannot start, report that
  rather than marking the DoD line green.
