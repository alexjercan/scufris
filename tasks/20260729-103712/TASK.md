# Extract the remaining routers and reduce create_app to assembly

- PRIORITY: 70
- TAGS: refactor, v0.2.0, backend, maintainability
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100441

## Story

As a maintainer, I want the remaining project, agent, chat, and diagnostics
routes in their own routers and `create_app` reduced to assembly, so that new
surfaces reuse one implementation instead of depending on a hand-synchronized
application factory.

## Steps

Each step is one atomic green commit: move the code, repoint the importers named
in it, add that router's slice of `tests/test_orchestrator_routers.py`, and run
`python -m pytest`. Routes keep their paths, methods, response models, status
codes and docstrings verbatim - this is a move, not a rewrite.

- [x] Add `scufris/env_bridge.py` (`ensure_api_base`, `ensure_den_path`, moved
      verbatim out of `app.py`) and `scufris/api/openapi.py`
      (`API_DESCRIPTION`, `OPENAPI_TAGS`, `route_tags`, and an
      `apply_route_tags(app)` wrapping the `iter_api_routes` tagging loop).
      Repoint `scufris/app.py`, `tests/test_env_isolation.py` and
      `tests/test_app.py`.
- [x] Extract `scufris/api/projects.py`: `ProjectDeps` + `build_project_router`
      over `/api/projects`, `/api/projects/discovered`, `/api/projects/new`,
      `/api/projects/{id}`, `/api/projects/{id}/tasks` and the
      `/projects/{id}` + `/projects/{id}/{rest:path}` SPA shells. Deps:
      `ProjectStore`, `Settings` (`project_base_dirs`, `web_dist`). Moves the
      `ProjectCreate`/`ProjectNew`/`ProjectUpdate`/`DiscoveredProject(s)` models.
- [x] Extract `scufris/api/agents.py`: `AgentDeps` + `build_agent_router` over
      `/api/agents` (list, create), `/api/agents/backends`,
      `/api/agents/pending`, `/api/agents/{id}` (get, patch, delete),
      `/api/agents/{id}/capabilities` and the `/agents/{id}` SPA shells. Deps:
      `AgentStore`, `SettingsStore`, `AgentRunService`, `Settings`. Carries
      `_update_orchestrator` and the `AgentCreate`/`AgentUpdate`/
      `BackendOption`/`PendingAgent`/`DeleteResult` models.
- [x] Extract `scufris/api/agent_runs.py`: `AgentRunDeps` +
      `build_agent_run_router` over the remaining `/api/agents/{id}/*` -
      `run`, `cancel`, `status`, `events`, `chat`, `request_input`,
      `report_back`, `acknowledge`, `fork`, `transcript`, `usage`, `memory`,
      `health`, `account`, `tools`, `mcp`. Deps: `AgentRunService`,
      `AgentDiagnostics`, `SessionGate`, `ReasoningStore`, `Settings`. Carries
      the `_require_agent*` / `_launch` / `_drain_turn` translators and the
      run/chat/signal models.
- [x] Extract `scufris/api/chat.py`: `ChatDeps` + `build_chat_router` over
      `/api/chat`, `/api/chat/stream`, `/api/chat/reset`, plus
      `_write_image_to_temp` and the `ImageAttachment`/`ChatRequest` models.
      Deps: `OrchestratorTurnService`, `Settings`.
- [x] Extract `scufris/api/legacy_agent.py`: `LegacyAgentDeps` +
      `build_legacy_agent_router` over every `/api/agent/*` - `info`, `config`
      (get/patch), `tools`, `tools/{name}/run`, `mcp`, `health`, `sessions`,
      `session` (post/fork/get/delete), `context`, `usage`, `memory`,
      `account`. Deps: `AgentStore`, `SettingsStore`, `AgentDiagnostics`,
      `AgentRunService`, the agent supervisor, `Settings`, the machine
      `api_token`. Repoint the `get_backend` bind-site list in
      `tests/conftest.py` and the same patch in `examples/host_agent.py`, and
      the `AgentConfigUpdate` import in `tests/test_settings_store.py`.
- [x] Extract the two non-route blocks that `create_app` cannot keep under the
      cap: `scufris/host_watch.py` (`HostWatchService` - one scheduled pass:
      run the checks, render, deliver, escalate a breach, plus
      `SCHEDULED_CHECK_ACTOR`) and `scufris/host_approval_bridge.py`
      (`HostApprovalBridge` - mark the requesting agent BLOCKED, deliver or
      defer a decision, drain on run completion, announce to Telegram).
- [x] Extract `scufris/telegram/wiring.py`: `build_approval_ops`,
      `build_settings_ops`, `start_bot`. Repoint the
      `monkeypatch.setattr("scufris.app.TelegramBot", ...)` calls in
      `tests/test_telegram_app.py` and `tests/test_auth_boundary.py`.
- [x] Reduce `create_app` to settings/collector defaults, the object graph, the
      lifespan, the two middlewares, the `include_router` calls, the static
      mount and `apply_route_tags`; remove `scufris/app.py` from `ALLOWLIST` in
      `scripts/check_file_size.py` and land
      `test_application_factory_assembles_domain_routers`.
- [x] Update `scufris/README.md` section 7 (the `api/` table and the "app.py
      still holds ..." paragraph) and section 8 (the module map rows for
      `app.py`, `api/`, the new `host_watch`/`host_approval_bridge`/
      `env_bridge` modules and `telegram/`) to the shipped layout.

## Definition of Done

- No route is registered on the application object itself, and the served route
  table - path, methods, response model, schema visibility and OpenAPI tag - is
  byte-identical to the recorded baseline
  (test: `test_application_factory_assembles_domain_routers`).
- The five new routers are drivable on a bare `FastAPI()` over fakes,
  constructing no `Settings`, store or `Database`
  (test: `test_orchestrator_routers_reach_for_nothing`,
  `test_the_agent_run_router_reaches_for_nothing`,
  `test_the_chat_router_reaches_for_nothing`,
  `test_the_legacy_agent_router_reaches_for_nothing`). Four tests, not one: the
  shared test file hit its line cap, so the claim is carried per router file.
- `/api/agent/*` answers out of the same diagnostics, turn and run services as
  the scoped `/api/agents/{id}/*` routes, with no second reader
  (test: `test_legacy_agent_routes_delegate_to_scoped_services`).
- `scufris/app.py` is inside the 600-line source cap with no allowlist entry
  (cmd: `python scripts/check_file_size.py`).
- The route table, middleware order, `app.state` keys and the five `create_app`
  override points are unchanged (cmd: `python -m pytest
  tests/test_route_contract.py`). The extraction's gate, not a red proof: it is
  green on base by construction, and any drift the move introduces turns it red.
- The suites pass with no drift
  (cmd: `python -m pytest && ruff check . && mypy .`).

## Progress

Checkpoint, 2026-08-03 (second). Five Steps landed as atomic green commits on
`refactor/extract-remaining-routers`:

- `4796ce1` env bridge + OpenAPI tag map
- `f76bff4` project router
- `ba0bf21` agent run router
- `633feb1` agent record router
- `24b4b0a` orchestrator chat router

The run router landed BEFORE the agent record router (Step 3), out of the
written order. Nothing depended on the order: the record surface imports the
`require_agent` / `require_agent_project` translators `api/agent_runs.py`
exports, which is the direction the plan wanted anyway.

Next Step: extract `scufris/api/legacy_agent.py` (Step 6, the `/api/agent/*`
surface). It is the largest single Step left - 16 routes plus the
`get_backend` / `AgentConfigUpdate` bind sites named in it - and it is
checkpointed BEFORE rather than mid-way for that reason. Then host_watch +
host_approval_bridge, telegram wiring, the `create_app` reduction and the
README sweep.

Verification at this checkpoint: `python -m pytest` (exit 0, 1075 passed 1
skipped), `ruff check .`, `mypy .` (220 files) and
`python scripts/check_file_size.py` all green. Tree clean. `scufris/app.py` is
1504 lines, down from 2621 on base, still on the ALLOWLIST until the last Step.

Diagnosis worth keeping: `tests/test_orchestrator_routers.py` HUNG the whole
suite at `test_a_live_host_approval_refuses_the_orchestrator_but_not_the
_operator`. The chat route answers with `relay_bus_sse(bus)`, and
`EventBus.subscribe` only ends when the bus closes - a fake `launch` that hands
back a bus nobody closes makes `TestClient.post` block forever, with no
failure, no timeout and no output. Found by `kill -SIGABRT` on the stuck pytest
and reading the faulthandler dump. Every fake `launch` now publishes a
`StreamDone` and closes its bus.

### Close-out, 2026-08-03

**What.** All ten Steps landed on `refactor/extract-remaining-routers` as atomic
green commits. `scufris/app.py` went from 2621 lines to 586, off the
`check_file_size.py` ALLOWLIST, holding no route: five new routers under `api/`
(`projects`, `agents`, `agent_runs`, `legacy_agent`, `chat`) joined the three
that already existed, and five non-route modules came out with them
(`env_bridge`, `api/openapi`, `host_watch`, `host_approval_bridge`,
`telegram/wiring`, plus `api/request_log` and `api/static` in the last Step).
`create_app` is now settings and collector defaults, the object graph, the
lifespan, the two middlewares, the `include_router` calls, the static mount and
the tag pass.

**Why this shape.** A router factory binds its dependencies at CONSTRUCTION over
a frozen deps dataclass, so a missing collaborator is a construction error at
the `include_router` line rather than a `NameError` on a live request - the old
routes read `scheduler` and `digests` out of a scope bound a thousand lines
below them and only worked because no request arrived before the factory
returned. Rejected alternatives are in `DECISION.md`.

**Difficulties.** Three worth keeping, all recorded above in full: the
`EventBus.subscribe` hang that froze the whole suite with no output (a fake
`launch` returning a bus nobody closes); `tests/test_orchestrator_routers.py`
hitting its 900-line test cap mid-task, which split the remaining coverage into
`test_chat_router.py` and `test_legacy_agent_router.py`; and the final
measurement landing 93 lines above the plan's projection, which took two HTTP
concerns out of the factory beyond the fallback the plan had named.

**Evidence.** `python -m pytest` exit 0 (1098 passed, 1 skipped),
`ruff check .`, `ruff format --check .` (227 files), `mypy .` (227 files),
`python scripts/check_file_size.py` and `python -m pytest
tests/test_route_contract.py` (5 passed) all green. The three DoD tests pass by
name. Tree clean.

**Reflection.** The line projection in the plan was built from measured segments
of `app.py` and was still 93 lines optimistic, because a move is not free: each
extraction leaves an `include_router` call, a deps construction and the comment
that explains it. Next time, budget ~10 lines of call site per extracted router
rather than counting only the lines that leave. The one bind site the Steps
missed was a logger NAME in a `caplog` assertion - grep for `scufris.<module>`
string literals as well as for imports when moving code between modules.

### Round 1 fixes, 2026-08-03

Six review findings answered in `bc769c2`, none disputed. One MAJOR: the run
router - the largest of the five, 16 routes - was the one new router with a rig
but no `_forbidden` pass, so DoD 2's claim covered four of five. It now has
`tests/test_agent_run_router.py`, the FOURTH file in this family, because
`test_orchestrator_routers.py` sits at 897 of its 900-line cap; the rig imports
that file's fakes rather than copying them, and redefines only the diagnostics
fake, since the run surface also asks for `health`/`tools`/`mcp`.

The five smaller ones were two stale docstrings, a stale largest-file claim, the
machine token rendering in a `LegacyAgentDeps` repr, and a zero-argument
middleware factory that bound nothing (`request_log_middleware()` flattened to a
module-level `log_requests`; `EXPECTED_MIDDLEWARE` already named the closure, so
the contract did not move).

Reflection to carry: this task ran out of test-file room three times. A move
that extracts N routers needs N test files budgeted from the start, not one
shared file that splits under pressure - the third split was forced by a review
finding rather than by the plan.

Turned up and NOT fixed here: a load-dependent flake in
`tests/test_app.py::_wait_state`, which polls a background run for 2s and then
returns the last state instead of failing. Neither the test nor the helper is
touched by this branch; recorded as task 20260803-100411.

## Notes

- Epic: 20260729-102145. Lane D, last task. Module split and its rejected
  alternatives: this task's `DECISION.md`.
- Proofs checked on the base branch. DoD 1: `create_app` registers 55 `APIRoute`
  objects directly on `app.router`, so the assertion is red today. DoD 2 and 3:
  the named tests and `tests/test_orchestrator_routers.py` do not exist. DoD 4:
  `python scripts/check_file_size.py` passes on base ONLY because of the
  allowlist entry - with the entry dropped it reports
  `scufris/app.py: 2621 lines, cap 600`, and with the file split but the entry
  kept it reports a stale entry, so the pair can only go green together.
- Measured on the base branch: `scufris/app.py` is 2621 lines against a 600-line
  source cap, so it is the ALLOWLIST ratchet entry this task removes. The guard
  is one-way - an entry that becomes stale fails the check - so the removal and
  the split must land in the same commit.
- Moving only the routes does NOT reach the cap. Measured segments of `app.py`:
  routes and their models ~1420 lines, the scheduled-checks block (860-989)
  ~130, the approval->agent bridge (1479-1626) ~148, the Telegram wiring
  (1628-1821) ~194, the OpenAPI tag map (180-263) ~85. `create_app` keeps its
  object graph, lifespan and middleware (~460 after the checks block leaves),
  which is why the last three Steps move non-route code too.
- Projected `app.py` after the split is ~575 lines, ~25 under the cap. If the
  final measurement lands over, the next thing out is `run_server` +
  `_ensure_api_base` into `cli.py` (its only caller) - do that rather than
  shaving comments.
- Measured, not projected: after the eight route/service Steps `app.py` was 668
  lines, 68 over the cap and 93 over that projection. The named fallback
  (`run_server` -> `cli.py`, ~30 lines with its now-unused imports) does not
  reach 600 on its own, so two more non-route blocks left with it, both HTTP
  concerns that were only in the factory because they were written there:
  `scufris/api/request_log.py` (the `log_requests` middleware; NOT `logsetup`,
  which the MCP servers and agent subprocesses import and which must stay free
  of a web stack) and `scufris/api/static.py` (`NoCacheStaticFiles` plus
  `mount_web_dist`, the `web_dist` presence branch and its warning). Final:
  586 lines, 14 under. `_propose_activation` was considered and KEPT: moving it
  into `hostconfig/` would make that package import `host_approvals`, a
  dependency edge the wiring exists to avoid.
- `tests/test_domain_routers.py` is 893 lines against the 900-line TEST cap, so
  the new coverage goes in a NEW `tests/test_orchestrator_routers.py` rather
  than extending it. Its `Rig`/`_forbidden` fixtures are the pattern to copy.
- Correction to the Steps, found while writing the agent-record slice:
  `tests/test_orchestrator_routers.py` reached 899/900 lines carrying the
  project, agent-run and agent-record slices, so the remaining slices CANNOT go
  there as the Steps say. One file per remaining router instead, same rig
  pattern: `tests/test_chat_router.py` (landed) and
  `tests/test_legacy_agent_router.py` (Step 6). No claim changes - only where
  it is asserted.
- `tests/test_route_contract.py` is the standing characterization baseline
  (route table, middleware order, `app.state` keys, the five `create_app`
  override points). It is green on the base branch by design, so it is a
  regression guard here and not a DoD proof; `EXPECTED_ROUTES` is what DoD 1
  asserts against and must not be edited.
- One bind site the Steps did not name, found by the suite rather than by grep:
  `tests/test_app.py::test_requests_are_logged` pins the middleware's LOGGER
  name (`caplog.at_level(..., logger="scufris.app")`), which the move to
  `api/request_log.py` changes. Repointed. `run_server`'s callers were safe:
  `tests/test_cli.py` patches `cli.run_server`, which is now where it is
  defined.
- Bind sites that break silently if missed, all found by grep and each named in
  its Step: `scufris.app.get_backend` (`tests/conftest.py` keeps the list;
  `examples/host_agent.py` patches it directly), `scufris.app.TelegramBot`
  (`tests/test_telegram_app.py`, `tests/test_auth_boundary.py`),
  `scufris.app._ensure_den_path` (`tests/test_env_isolation.py`,
  `tests/test_app.py`), `scufris.app.AgentConfigUpdate`
  (`tests/test_settings_store.py`).
- Response models stay at MODULE level in each router module. A model defined
  inside a router factory is a local whose forward reference FastAPI cannot
  resolve, and `app.openapi()` fails with "not fully defined" - see the
  `DigestView` comment in `scufris/api/host.py`.
- Routers are included BARE (absolute paths on the router, no `prefix=`, no
  `tags=`). `api/routes.py::iter_routes` raises on a prefixed or tagged
  include, and `route_tags` assigns by path.
- Refactor only. No new product behavior, no changed status codes, no new
  settings.
