# Review: Extract the remaining routers and reduce create_app to assembly

- TASK: 20260729-103712
- BRANCH: refactor/extract-remaining-routers

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) tests/test_orchestrator_routers.py:277 - DoD 2 claims "the
  five new routers are drivable on a bare `FastAPI()` over fakes, constructing
  no `Settings`, store or `Database`", but only four carry the booby trap:
  `test_orchestrator_routers_reach_for_nothing` drives the project and
  agent-record routers, `test_the_chat_router_reaches_for_nothing`
  (`tests/test_chat_router.py:221`) the chat one, and
  `test_the_legacy_agent_router_reaches_for_nothing`
  (`tests/test_legacy_agent_router.py:512`) the legacy one. `api/agent_runs.py`
  - the largest new router, 541 lines over 16 routes - has an `AgentRunRig` on
  a bare app but no `_forbidden` pass, so a `Settings()` or `ProjectStore(...)`
  creeping back into a run-route body is unpinned. Drive the run surface
  (`/status`, `/transcript`, `/usage`, `/memory`, `/health`, `/account`,
  `/tools`, `/mcp`, `POST /run`) under the same traps.
  `tests/test_orchestrator_routers.py` is at 897/900, so this needs its own
  file - `tests/test_agent_run_router.py` - not an extension there.
  - Response: fixed in bc769c2. `tests/test_agent_run_router.py` drives
    `/status`, `/transcript`, `/usage`, `/memory`, `/health`, `/account`,
    `/tools`, `/mcp`, `POST /run`, `/cancel`, `/request_input`, `/report_back`,
    `/acknowledge` and `/chat` under the same four `__init__` traps plus
    `state_database`. `/events` is the one route left out and the docstring
    says why: it relays a live bus, so it blocks forever over a fake. The rig
    imports the fakes from `test_orchestrator_routers` rather than copying
    them; only the diagnostics fake is redefined, because the run surface asks
    for `health`/`tools`/`mcp` too and the shared one answers three methods.
    Checked red the way the other traps were: a `Settings()` inserted at the
    top of `agent_run_status` turns it red with the trap's own message.

- [x] R1.2 (MINOR) scufris/env_bridge.py:11 - the module docstring's
  justification is wrong about its own callers: it says `ensure_den_path` "is
  called from three routers and the Telegram wiring as well as from
  `run_server`". `ensure_den_path` is called from two routers
  (`api/agent_runs.py`, `api/legacy_agent.py`) and `telegram/wiring.py`;
  `run_server` calls `ensure_api_base`. Reword to "called from two routers and
  the Telegram wiring, and `ensure_api_base` from `run_server`".
  - Response: fixed in bc769c2, in the reviewer's words.

- [x] R1.3 (MINOR) tests/test_app.py:76 - `FakeBackend`'s docstring still says
  it is "Injected by monkeypatching ``scufris.app.get_backend``". That symbol
  no longer exists; the fixture now patches `scufris.api.agent_runs`,
  `scufris.api.legacy_agent` and `scufris.orchestrator.runs`. Repoint it to
  `tests/conftest.py::patch_get_backend`.
  - Response: fixed in bc769c2.

- [x] R1.4 (NIT) scufris/api/legacy_agent.py:168 - `api_token: str` sits in a
  frozen dataclass with the default `repr`, so the machine credential renders
  in any traceback that prints `LegacyAgentDeps` (`pytest --showlocals` over
  `create_app`, a `TypeError` at the `build_legacy_agent_router` call).
  Declare it `api_token: str = field(repr=False)` and import `field`.
  - Response: fixed in bc769c2. The deps docstring now says why the field is
    `repr=False`, so the next reader does not drop it as noise.

- [x] R1.5 (NIT) scufris/api/request_log.py:26 - `request_log_middleware()` is
  a zero-argument factory that binds nothing and returns the closure defined
  inside it, with one caller. Delete the wrapper and its `return log_requests`,
  dedent `log_requests` to module level, and call
  `app.middleware("http")(log_requests)` in `create_app`. (`auth_middleware`
  stays a factory - it actually binds arguments.)
  - Response: fixed in bc769c2. The now-unused `Dispatch` import went with it.
    `test_route_contract.py::EXPECTED_MIDDLEWARE` already named `log_requests`
    (the closure's own name), so the middleware contract is unchanged.

- [x] R1.6 (NIT) tests/test_check_file_size.py:55 - the docstring claims
  `web/src/style.css` "is the largest file in the tree after
  `scufris/app.py`". `scufris/app.py` is 586 lines now; the largest is
  `tests/test_app.py`. Reword to "after `tests/test_app.py`".
  - Response: fixed in bc769c2.

Verification notes:

- All six DoD proofs run in the worktree by both the out-of-context reviewer
  and the recording pass: `python scripts/check_file_size.py` exit 0 with
  `scufris/app.py` off the ALLOWLIST at 586 lines; `python -m pytest` green
  (1099 collected, 1 skipped); `python -m pytest tests/test_route_contract.py`
  5 passed; `ruff check .` clean; `mypy .` clean over 227 files. No `manual:`
  proofs in this task.
- The move was checked as a move: every top-level and nested `def`/`class` in
  `git show master:scufris/app.py` was extracted and diffed against its landing
  site. All 55 route bodies, response models, both env bridges, `run_server`,
  `log_requests`, `_NoCacheStaticFiles`, `route_tags`/`OPENAPI_TAGS`/
  `API_DESCRIPTION`, the scheduled-check block and the approval-bridge block
  (including hook order and the completion fan-out order) are verbatim modulo
  the mechanical `deps.` / `self._` rewrites and `ruff format` rewrapping.
- Route count 55 -> 55, with relative order preserved inside every shadowing
  family (`/api/projects/discovered` and `/new` before `{project_id}`;
  `/api/agents/backends` and `/pending` before `{agent_id}`;
  `/api/agent/session/fork` before `/session/{id}`; `/api/agent/tools` before
  `/tools/{name}/run`). Cross-router include order cannot shadow - no `{param}`
  spans a `/`. Worth knowing: `test_route_contract.py::_route_table` is
  `sorted()`, so it does NOT pin registration order; that was checked by hand.
- `EXPECTED_ROUTES` and `EXPECTED_MIDDLEWARE` are unedited; the only change to
  `tests/test_route_contract.py` is the added
  `test_application_factory_assembles_domain_routers`.
- Bind sites repointed and confirmed: `get_backend` (the `conftest.py` list and
  `examples/host_agent.py`), `TelegramBot` (two tests), `_ensure_den_path` (two
  tests), `AgentConfigUpdate`, and the `caplog` logger name in
  `test_requests_are_logged`. No `raising=False` patches hiding a stale target.
- The recording pass independently re-derived R1.1 (grepped every `_forbidden`
  site across the three new test files against the five router modules) and
  R1.2 (grepped every `ensure_den_path` / `ensure_api_base` caller).
- No existing test was deleted or weakened; `test_host_digest.py` and
  `test_route_contract.py` are additive and the rest of the test diff is symbol
  repointing.
- Not verified: the Telegram poll loop and the scheduled-check pass against a
  real bot or real host - both are exercised only through fakes.

Process signal: `tests/test_orchestrator_routers.py` landed at 897 of the
900-line TEST cap, so the R1.1 coverage needs a fourth test file. That is the
third mid-flight file split on this task - the Steps assumed one test file for
all six routers, and the plan's own correction note already recorded two.

## Fix notes for round 1

All six findings answered in `bc769c2`; none was pushed back on. Re-verified
after the fixes, not just around them: `python -m pytest` (1099 passed, 1
skipped), `ruff check .`, `ruff format --check .` (228 files), `mypy .` (228
files), `python scripts/check_file_size.py` and `python -m pytest
tests/test_route_contract.py` all green, tree clean.

One thing the re-run turned up that is NOT this branch's:
`tests/test_app.py::test_orchestrator_chat_uses_server_cwd` failed in one
full-suite run and passed in the two full-suite runs and eight isolated runs
either side of it, with no code change between. `_wait_state` polls a
background run 200 times at 10ms and then RETURNS the last state instead of
failing, so a lapsed deadline surfaces as a confusing assertion about a session
id. Neither the test nor the helper is touched by this branch
(`git log master..HEAD -- tests/test_app.py` is the env-bridge and
factory-reduction commits, neither of which mentions `_wait_state`). Recorded
as task 20260803-100411 under `work/review-feedback.md` section 6.

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

- [ ] R2.1 (MINOR) tests/test_agent_run_router.py:152 - the trap docstring says
  "``/events`` is the one route left out", and R1.1's Response repeats it, but
  `POST /api/agents/{agent_id}/fork` (`scufris/api/agent_runs.py:416`) is also
  undriven: the trap drives 14 of the router's 16 routes. `/fork` does not
  block the way `/events` does - it ends in `launch` + `relay_bus_sse`, exactly
  like `/chat`, which the same test drives green. Either add a
  `fork_seed(self, agent, session_id, message_index, text) -> str` to
  `FakeRunService` (`tests/test_orchestrator_routers.py:343`) and a
  `trap_client.post(f"/api/agents/{AGENT_ID}/fork", json={"message_index": 0,
  "text": "go"})` assertion, or reword the docstring to name `/fork` as the
  second exclusion with its reason.
  - Response:

- [ ] R2.2 (MINOR) scufris/README.md:85 - the trust-boundary table still points
  the Telegram chat-id allowlist re-check at `app._build_telegram_approval_ops`.
  This branch moved that function to
  `scufris/telegram/wiring.py::build_approval_ops` (`scufris/app.py:555`), so
  the document `AGENTS.md` names as the source of truth for trust boundaries
  cites a symbol that no longer exists. Replace
  `app._build_telegram_approval_ops` with
  `telegram/wiring.py::build_approval_ops`.
  - Response:

- [ ] R2.3 (NIT) scufris/host_approval_bridge.py:26 -
  `logger = logging.getLogger(__name__)` came across in the extraction but the
  module never logs (0 `logger.` uses; `host_watch.py` has 4,
  `telegram/wiring.py` 2). Delete line 26 and the `import logging` on line 17.
  - Response:

Round-1 fixes, each verified against the branch rather than against its
Response:

- R1.1 confirmed. `tests/test_agent_run_router.py` exists and drives every
  route the finding named - `/status`, `/transcript`, `/usage`, `/memory`,
  `/health`, `/account`, `/tools`, `/mcp`, `POST /run` - plus `/cancel`,
  `/request_input`, `/report_back`, `/acknowledge` and `/chat`, under
  `__init__` traps on `Settings`, `AgentStore`, `ProjectStore`, `Database` and
  a trap on `scufris.db.state_database`. The requested change is delivered in
  full; the residual `/fork` gap is beyond what R1.1 asked and is carried as
  R2.1.
- R1.2 confirmed. `ensure_den_path` has exactly two router callers
  (`api/agent_runs.py`, `api/legacy_agent.py`) plus `telegram/wiring.py`;
  `ensure_api_base` has one, `cli.py:37`. The docstring now says that.
- R1.3 confirmed. `FakeBackend` points at `tests/conftest.py::patch_get_backend`.
- R1.4 confirmed. `api_token: str = field(repr=False)`, with the docstring
  saying why so it is not pruned as noise.
- R1.5 confirmed. The zero-argument factory and the `Dispatch` import are gone;
  `create_app` calls `app.middleware("http")(log_requests)` and
  `EXPECTED_MIDDLEWARE` is unchanged.
- R1.6 confirmed. `tests/test_app.py` (4242 lines) is the largest file;
  `web/src/style.css` (2662) is next.

Verification notes:

- Every DoD proof re-run in the worktree by both the out-of-context reviewer
  and this recording pass. `python -m pytest` green (1099 passed, 1 skipped,
  exit 0); `ruff check .` clean; `ruff format --check .` 228 files;
  `mypy .` clean over 228 files; `python scripts/check_file_size.py` exit 0
  with `scufris/app.py` at 586 lines and off the ALLOWLIST (which now holds
  `tests/test_app.py` alone); `python -m pytest tests/test_route_contract.py`
  5 passed; the six named proof tests pass by name (6 passed, 1094 deselected).
  No `manual:` proofs in this task, so there are no pending user checks.
- Load-bearing claims re-derived independently of the reviewer: R2.1 by
  listing all 16 `@router` decorators in `scufris/api/agent_runs.py` against
  every path the trap test drives - `/events` and `/fork` are the two
  survivors, not one; R2.2 by grepping `_build_telegram_approval_ops` across
  the tree, which hits `scufris/README.md:85` and one `tasks/` record (exempt
  as append-only history); R2.3 by counting `logger.` uses in
  `scufris/host_approval_bridge.py`, which is zero.
- The fixes introduced no regression: the round-1 commit `bc769c2` touches
  only the six finding sites, and the full suite, formatter, linter and type
  checker are green after it, not merely around it.
- All three round-2 findings are MINOR or NIT, so none blocks the verdict.
- Not verified, unchanged from round 1: the Telegram poll loop and the
  scheduled-check pass against a real bot or real host, both exercised only
  through fakes. The claimed red-check of the new trap was not reproduced -
  doing so requires editing a source file.
