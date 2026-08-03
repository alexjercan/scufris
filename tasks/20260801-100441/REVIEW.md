# Review: Extract the orchestrator-turn and agent-run services

- TASK: 20260801-100441
- BRANCH: refactor/orchestrator-services

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) scufris/orchestrator/runs.py:145 - `require_agent_async`
  (runs.py:145-152) and `require_agent_project_async` (runs.py:169-179) have
  zero callers in `scufris/`, `tests/` or `scripts/`. `app.py` keeps its own
  `_require_agent_async` / `_require_agent_project_async` closures
  (app.py:1441, app.py:1455), which offload the SYNCHRONOUS service methods so
  they can also translate `AgentNotFound` / `AgentProjectMissing` into
  `HTTPException`; only `require_agent` and `require_agent_project` are ever
  reached. Two service methods no requirement names. Delete runs.py:145-152 and
  runs.py:169-179, moving the offload rationale from
  `require_agent_project_async`'s docstring onto `require_agent_project`.
  - Response: Fixed. Both async wrappers are deleted; the offload rationale (the
    immediate-begin write lock, `scufris/db/engine.py`) now sits on
    `require_agent_project`'s own docstring, telling loop-thread callers to
    offload it. `app.py` keeps its own closures because they must also translate
    to `HTTPException`.
- [x] R1.2 (MAJOR) scufris/orchestrator/turn.py:129 - `OrchestratorTurnService.cancel`
  is exercised by no test against the real service. Its only production caller
  is the Telegram `/cancel` callback, and `tests/test_telegram_app.py:293`
  (`test_on_cancel_false_when_idle`) now asserts `_FakeTurn(cancelled=False)`
  returns `False` - a fake echoing its own constructor argument. On `master`
  that test drove the real callback over a real supervisor plus `active_run_id`
  and so pinned "an idle orchestrator cancels to False"; the invariant moved
  into `turn.py:135-139` (`except NoActiveRun: return False`) and was not
  re-pinned there. The web cancel path does NOT cover it: it reaches
  `runs.cancel` through `/api/agents/{id}/cancel` (app.py:1856), never
  `turn.cancel`. Add a case to `tests/test_orchestrator_service.py` building
  `OrchestratorTurnService` over the real `AgentRunService` from
  `_run_service`, asserting `await turn.cancel() is False` when idle and
  `is True` after a live `runs.launch(orchestrator, ...)`.
  - Response: Fixed. `test_the_turn_service_cancels_only_a_live_orchestrator_turn`
    builds `OrchestratorTurnService` over the real `AgentRunService` and real
    supervisor from `_run_service`, and pins False/idle then True/live (plus
    `busy()` on both sides). `_run_service` now also returns its supervisor and
    sets `agent_backend=mock` so the reserved orchestrator record resolves to the
    mock backend. Falsified: forcing `except NoActiveRun: return True` reds it.
- [x] R1.3 (MINOR) scufris/orchestrator/turn.py:116 - `wake`'s
  `except RunAlreadyActive: return False` branch is taken by no test;
  `test_orchestrator_transports_share_turn_service`
  (tests/test_orchestrator_service.py:175) drives only the granted path. This
  is the branch the wake bridge's whole back-off depends on. Extend that test,
  or add one, calling `turn.wake` against a `_RecordingRuns` whose `launch`
  raises `RunAlreadyActive`, asserting `False` and no extra entry in
  `runs.launched`.
  - Response: Fixed. `test_orchestrator_transports_share_turn_service` now ends
    by calling `turn.wake` over a `_BusyRuns` (a `_RecordingRuns` whose `launch`
    raises `RunAlreadyActive`), asserting `False` and `launched == []`.
    Falsified: flipping the branch to `return True` reds it.
- [x] R1.4 (MINOR) scufris/orchestrator/runs.py:98 - the moved comment gained a
  task-id fragment the original did not have, and the fragment is a literal
  placeholder: `(review round 1, R1.1 of 20260722-...)`. Same at runs.py:202,
  `(DECISION.md 2 of 20260722-...)`. Both read `(review round 1, R1.1)` and
  `(DECISION.md 2)` on `master` (app.py:757, app.py:1574). AGENTS.md:103: "Task
  IDs belong in task records and Markdown, never in code comments or
  docstrings." Delete the ` of 20260722-...` fragment from both lines.
  - Response: Fixed. Both lines read `(review round 1, R1.1)` and
    `(DECISION.md 2)` again, as on `master`.
- [x] R1.5 (MINOR) tasks/20260801-100441/TASK.md:50 - Step 4's literal text says
  "the WakeBridge's `launch`/`is_orchestrator_busy` become
  `runs.launch`/`runs.active`". They became `turn.wake`/`turn.busy`
  (app.py:1608-1609). The outcome is right - `runs.launch` has the wrong arity
  for `Callable[[str], Awaitable[bool]]` and the `ORCHESTRATOR_ID` lookup has
  to live somewhere - but the Step is ticked with the deviation unrecorded, and
  `turn.wake` is absent from the close-out's own list of the turn service's
  methods ("send/stream/reset/cancel/busy"). Add a "Discovered while working"
  bullet naming `turn.wake`/`turn.busy` and why, and add `wake` to the
  close-out list.
  - Response: Fixed. "Discovered while working" gains a bullet naming
    `turn.wake`/`turn.busy` and why (`runs.launch` has the wrong arity for
    `Callable[[str], Awaitable[bool]]` and raises where the bridge wants a
    False), and the close-out list now reads send/stream/wake/reset/cancel/busy.
- [x] R1.6 (MINOR) tasks/20260801-100441/TASK.md:65 - Step 5 says the turn
  service "owns the `settings.agent_enabled` check ... that the three
  transports each repeat today", but `post_chat_reset` (app.py:2526) still
  raises its own 503 and `OrchestratorTurnService.reset` (turn.py:119) has no
  gate. Deliberate - gating it would make the Telegram `/new` raise
  `AgentDisabled` out of an `on_reset` with no handler - but unrecorded. Note
  the reset exemption in the Step or in Notes, the way `/api/chat/stream`'s
  in-route image decode is already justified.
  - Response: Fixed. A "Discovered while working" bullet records the reset
    exemption and its reason - gating it would make the Telegram `/new` raise
    `AgentDisabled` out of an `on_reset` with no handler.
- [x] R1.7 (NIT) tests/test_app.py:3205 - three comments still name the deleted
  `_launch_agent_turn`: tests/test_app.py:3205, tests/test_app.py:3869 and
  tests/test_telegram_app.py:376. Replace with `AgentRunService.launch`.
  - Response: Fixed. All three now name `AgentRunService.launch`; no
    `_launch_agent_turn` remains outside `tasks/`.
- [x] R1.8 (NIT) tests/conftest.py:186 - the docstring is accurate about a bind
  site that DISAPPEARS (`monkeypatch.setattr` raises on a missing attribute),
  but the failure this helper was written for was a bind site that APPEARED and
  was not listed, which stays silent. Reword to "a listed bind site that
  disappears fails loudly; a new one must be added here", so the helper is not
  read as protection it does not give.
  - Response: Fixed. The docstring now says a LISTED bind site that disappears
    fails loudly and a NEW one is silent until added here, so the helper is not
    read as catching a caller it has never heard of.
- [x] R1.9 (NIT) tasks/20260801-100441/TASK.md:236 - the out-of-context reviewer
  saw `tests/test_host_action_api.py::test_cancelling_a_live_apply_is_recorded`
  go red once in three full-suite runs; the Evidence section pre-blames exactly
  two flakes and does not name it. The primary could NOT reproduce it (0 reds
  in 3 full-suite runs plus 5 isolated runs), and this diff does not touch that
  test, so it is recorded at NIT rather than dropped: add it to the flake list
  if it reappears, and do not treat one sighting as a regression.
  - Response: Acknowledged, no change. Not reproducible here (0 reds in this
    round's full-suite runs), and this diff does not touch the host-action apply
    path. Left off the Evidence flake list deliberately - pre-blaming a flake
    nobody can reproduce is how a real regression gets waved through.

Verified independently by the primary, not taken from the record:

- `ruff check .`, `ruff format --check .`, `mypy .` - clean, 211 files.
- `python -m pytest` - exit 0, zero failures, three consecutive full-suite runs.
  Neither flake the record pre-blames reproduced.
- `cd web && npm run ci` - exit 0.
- Every `cmd:` proof run by hand: the `test -d && ! rg` transport-import guard
  exits 0, `tests/test_route_contract.py` passes, `check_file_size.py` exits 0
  (`runs.py` 513, `turn.py` 142, `errors.py` 89, all under the 600 cap;
  `app.py` 2621, down from 2923).
- The SSE capture in `test_chat_stream_events_are_unchanged` is genuine, not
  self-confirming: the primary replayed both requests against `master` and both
  strings match byte-for-byte, padding and `id:` sequence included.
- The completion fan-out order is preserved exactly - `wake_bridge.on_run_complete`
  then `_drain_deferred_decision`, both after `mark_finished` and inside the
  supervisor's finally (runs.py:337-338 against master's app.py:1708-1713).
- The reset serialize-key invariant dropped from `tests/test_telegram_app.py` IS
  re-pinned at its new boundary (tests/test_orchestrator_service.py:216,
  `supervisor.serialized_keys == [ORCHESTRATOR_ID]`). R1.2 is the one invariant
  that was not.
- The Telegram refusal strings are unchanged from `master`, and the narrowing
  from `except HTTPException` to `except RunAlreadyActive` is the one the plan's
  Notes call for. A raising completion hook stays contained: the supervisor
  catches and logs it (supervisor.py:356-362).
- Doc sweep: no live doc, README, AGENTS.md or `web/` file mentions
  `_launch_agent_turn`, `_agent_run_active`, `_orchestrator_busy` or
  `_wake_launch`. Only `tasks/` (exempt) and the three test comments in R1.7.
- `tatr -r . check` - one pre-existing warning on 20260803-014401, untouched by
  this diff.
- No `manual:` proofs exist in this task, so there are no pending user checks.

Observations, not findings:

- `/api/chat/stream` leaks the image tempdir when the launch refuses: `cleanup`
  is wired only as `on_done`, which never fires if `turn.stream` raises.
  Identical on `master`, so it is a pre-existing defect and out of this diff's
  scope - worth its own task.
- Process signal: the branch appends a 40-line diagnosis to a SIBLING record,
  `tasks/20260803-043935/TASK.md`. The content is accurate and useful, but it is
  scope the Story did not name.

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

All nine round-1 findings verified fixed and ticked above. One new finding, a
cosmetic regression from the round-1 edits; NIT, so it does not block.

- [x] R2.1 (NIT) scufris/orchestrator/runs.py:189 - deleting the ` of
  20260722-...` fragment (R1.4) and retargeting the comment at
  `AgentRunService.launch` (R1.7) shortened a line in each paragraph without
  rewrapping the rest, leaving a ragged half-line: `launch`'s docstring breaks
  after "the guard and the claim are one" (runs.py:189-190) and
  tests/test_app.py:3869-3870 breaks after "so the sub-agent / run is".
  Rewrap both paragraphs to the 88-column margin.
  - Response: Fixed during close-out. Both paragraphs rewrapped to 88 columns;
    comment text only, no code change. `ruff check .`, `ruff format --check .`
    and `mypy .` re-run clean.

Verified independently by the primary, not taken from the reviewer's record:

- `ruff check .`, `ruff format --check .`, `mypy .`, `python -m pytest` - all
  exit 0. Neither pre-blamed flake reproduced.
- R1.2 re-derived by mutation, not by reading the test: flipping
  `except NoActiveRun: return False` to `return True` (turn.py:135-137) reds
  `test_the_turn_service_cancels_only_a_live_orchestrator_turn`. The invariant
  is genuinely pinned at its new boundary; the working tree was restored and is
  clean.
- R1.1 re-derived by grep: no `require_agent_async` /
  `require_agent_project_async` on the service. The only survivors are
  `app.py`'s own closures (app.py:1441, app.py:1455), which exist to translate
  to `HTTPException` - the Response's stated reason.
- R2.1 traced to its cause in `git show 8b1ba53`: both ragged lines are in that
  commit's hunks, so this is a round-1 fix regression rather than a
  pre-existing wrap.
- R1.4, R1.7, R1.8 read at their sites; all three match their Responses.
- No `manual:` proofs exist in this task, so there are no pending user checks.
- R1.9 stands as acknowledged-no-change. `test_cancelling_a_live_apply_is_recorded`
  did not red in this round either; not added to the flake list.
