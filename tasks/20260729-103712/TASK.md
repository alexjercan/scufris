# Extract the remaining routers and reduce create_app to assembly

- PRIORITY: 70
- TAGS: refactor, v0.2.0, backend, maintainability
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
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

- [ ] Add `scufris/env_bridge.py` (`ensure_api_base`, `ensure_den_path`, moved
      verbatim out of `app.py`) and `scufris/api/openapi.py`
      (`API_DESCRIPTION`, `OPENAPI_TAGS`, `route_tags`, and an
      `apply_route_tags(app)` wrapping the `iter_api_routes` tagging loop).
      Repoint `scufris/app.py`, `tests/test_env_isolation.py` and
      `tests/test_app.py`.
- [ ] Extract `scufris/api/projects.py`: `ProjectDeps` + `build_project_router`
      over `/api/projects`, `/api/projects/discovered`, `/api/projects/new`,
      `/api/projects/{id}`, `/api/projects/{id}/tasks` and the
      `/projects/{id}` + `/projects/{id}/{rest:path}` SPA shells. Deps:
      `ProjectStore`, `Settings` (`project_base_dirs`, `web_dist`). Moves the
      `ProjectCreate`/`ProjectNew`/`ProjectUpdate`/`DiscoveredProject(s)` models.
- [ ] Extract `scufris/api/agents.py`: `AgentDeps` + `build_agent_router` over
      `/api/agents` (list, create), `/api/agents/backends`,
      `/api/agents/pending`, `/api/agents/{id}` (get, patch, delete),
      `/api/agents/{id}/capabilities` and the `/agents/{id}` SPA shells. Deps:
      `AgentStore`, `SettingsStore`, `AgentRunService`, `Settings`. Carries
      `_update_orchestrator` and the `AgentCreate`/`AgentUpdate`/
      `BackendOption`/`PendingAgent`/`DeleteResult` models.
- [ ] Extract `scufris/api/agent_runs.py`: `AgentRunDeps` +
      `build_agent_run_router` over the remaining `/api/agents/{id}/*` -
      `run`, `cancel`, `status`, `events`, `chat`, `request_input`,
      `report_back`, `acknowledge`, `fork`, `transcript`, `usage`, `memory`,
      `health`, `account`, `tools`, `mcp`. Deps: `AgentRunService`,
      `AgentDiagnostics`, `SessionGate`, `ReasoningStore`, `Settings`. Carries
      the `_require_agent*` / `_launch` / `_drain_turn` translators and the
      run/chat/signal models.
- [ ] Extract `scufris/api/chat.py`: `ChatDeps` + `build_chat_router` over
      `/api/chat`, `/api/chat/stream`, `/api/chat/reset`, plus
      `_write_image_to_temp` and the `ImageAttachment`/`ChatRequest` models.
      Deps: `OrchestratorTurnService`, `Settings`.
- [ ] Extract `scufris/api/legacy_agent.py`: `LegacyAgentDeps` +
      `build_legacy_agent_router` over every `/api/agent/*` - `info`, `config`
      (get/patch), `tools`, `tools/{name}/run`, `mcp`, `health`, `sessions`,
      `session` (post/fork/get/delete), `context`, `usage`, `memory`,
      `account`. Deps: `AgentStore`, `SettingsStore`, `AgentDiagnostics`,
      `AgentRunService`, the agent supervisor, `Settings`, the machine
      `api_token`. Repoint the `get_backend` bind-site list in
      `tests/conftest.py` and the same patch in `examples/host_agent.py`, and
      the `AgentConfigUpdate` import in `tests/test_settings_store.py`.
- [ ] Extract the two non-route blocks that `create_app` cannot keep under the
      cap: `scufris/host_watch.py` (`HostWatchService` - one scheduled pass:
      run the checks, render, deliver, escalate a breach, plus
      `SCHEDULED_CHECK_ACTOR`) and `scufris/host_approval_bridge.py`
      (`HostApprovalBridge` - mark the requesting agent BLOCKED, deliver or
      defer a decision, drain on run completion, announce to Telegram).
- [ ] Extract `scufris/telegram/wiring.py`: `build_approval_ops`,
      `build_settings_ops`, `start_bot`. Repoint the
      `monkeypatch.setattr("scufris.app.TelegramBot", ...)` calls in
      `tests/test_telegram_app.py` and `tests/test_auth_boundary.py`.
- [ ] Reduce `create_app` to settings/collector defaults, the object graph, the
      lifespan, the two middlewares, the `include_router` calls, the static
      mount and `apply_route_tags`; remove `scufris/app.py` from `ALLOWLIST` in
      `scripts/check_file_size.py` and land
      `test_application_factory_assembles_domain_routers`.
- [ ] Update `scufris/README.md` section 7 (the `api/` table and the "app.py
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
  (test: `test_orchestrator_routers_reach_for_nothing`).
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
- `tests/test_domain_routers.py` is 893 lines against the 900-line TEST cap, so
  the new coverage goes in a NEW `tests/test_orchestrator_routers.py` rather
  than extending it. Its `Rig`/`_forbidden` fixtures are the pattern to copy.
- `tests/test_route_contract.py` is the standing characterization baseline
  (route table, middleware order, `app.state` keys, the five `create_app`
  override points). It is green on the base branch by design, so it is a
  regression guard here and not a DoD proof; `EXPECTED_ROUTES` is what DoD 1
  asserts against and must not be edited.
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
