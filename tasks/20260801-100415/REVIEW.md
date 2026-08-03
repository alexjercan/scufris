# Review: Delegate legacy /api/agent/* routes to orchestrator diagnostics

- TASK: 20260801-100415
- BRANCH: fix/legacy-agent-diagnostics

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) web/src/agent-view.ts:145 - the only frontend consumer of the
  BREAKING `/api/agent/usage` shape change is untested. The jsdom fetch stub in
  `web/src/agent-view.test.ts:104` (and the twin at :247) is a catch-all
  returning `{}` for every non-`/tools` url, so `quota.value` is `undefined` and
  nothing fails if the unwrap is dropped and the raw envelope is handed to
  `renderUsage`. DECISION.md names this call site as the ENTIRE cost of the
  chosen alternative, so it is the one thing the alternative owes a pin. Add a
  case in `agent-view.test.ts` that answers `/api/agent/usage` with
  `{"supported": true, "value": <quota>}` and asserts the meter renders
  populated, plus one answering `{"supported": false, "value": null}` asserting
  it stays hidden.
  - Response: fixed. `web/src/agent-view.test.ts` gains a
    "legacy /api/agent/usage capability envelope (startAgent)" describe with
    both cases, mounting `startAgent` over a `#usage-meter` and a fetch stub
    that answers `/api/agent/usage` with the envelope. Pinned red at its own
    boundary: replacing `renderUsage(quota.value)` with
    `renderUsage(quota as unknown as UsageQuota)` fails the supported case
    (`meter.hidden` true, expected false). Restored; 7 tests green.

- [x] R1.2 (MINOR) tests/test_app.py:816 - the new codex-backed client in
  `test_agent_health_endpoint_reports_checks` sets `agent_enabled=True` but
  pins no `codex_home`, so `scufris/health.py:258` resolves
  `Path.home() / ".codex"` and the test reads the DEVELOPER's real codex home
  for its session-count check. This is the exact hazard the Close-out records
  fixing for the three disabled-agent tests. Add
  `codex_home=tmp_path / "no-codex"` to that `Settings(...)`.
  - Response: fixed, and widened. The MOCK-backed client above it is
    `agent_enabled=True` too, and `scufris/health.py:258` gates on
    `agent_enabled` alone, not on the backend - so it read the real home by the
    same path. Both `Settings(...)` in the test now pin
    `codex_home=tmp_path / "no-codex"`.

- [x] R1.3 (MINOR) tests/test_app.py:1852 and tests/test_app.py:1908 -
  `test_usage_empty_reading_when_disabled` and `test_memory_zero_when_disabled`
  now pin an empty `codex_home` and assert `supported: true` with an empty
  reading, which is what `test_memory_endpoint_empty_ok` (:1889) already
  asserts for an ENABLED agent. Nothing in either test depends on
  `agent_enabled=False` any more, so DECISION-4 (the short-circuit removal:
  disabled is not unsupported) is asserted by no test. Assert the
  disabled-specific property in the same app - `/api/agent/account` reporting
  `enabled: false` beside a `supported: true` quota - or drop the now-redundant
  cases.
  - Response: fixed by the first branch. The two cases collapse into one
    `test_disabled_agent_is_supported_not_unsupported`, which drives usage,
    memory AND account off the SAME disabled app and asserts the
    disabled-specific property: `enabled: false` beside a
    `quota: {"supported": true, "value": null}`. `test_memory_zero_when_disabled`
    is deleted as redundant with `test_memory_endpoint_empty_ok`. Suite is 981,
    not 982, by that one deletion.

- [x] R1.4 (NIT) scufris/app.py:3441 - the retained docstring opener "Read-only
  diagnostics for the operator console (never raises)" now sits above
  `await _require_agent_async(ORCHESTRATOR_ID)`, which raises `HTTPException`
  404 on `AgentNotFound`. It cannot fire for the synthetic orchestrator, but the
  claim is no longer literally true. Reword to "never raises for the
  orchestrator (its record is synthetic)".
  - Response: fixed. `scufris/app.py:3439` now opens "Read-only diagnostics for
    the operator console (never raises for the orchestrator, whose record is
    synthetic)".

- [x] R1.5 (NIT) scufris/app.py:2751 and scufris/app.py:2882 - both new
  orchestrator lookups call `agents.get` under `asyncio.to_thread` rather than
  the established `_require_agent_async` helper every other call site uses.
  Correct (an `HTTPException` off the HTTP path would be wrong) but
  undocumented. Add a one-line comment saying so, at :2751, so the next reader
  does not "fix" it.
  - Response: fixed at the digest site (`scufris/app.py:2750`), which is the
    one the finding asks for: "`agents.get`, not `_require_agent_async`: this
    runs off the HTTP path, where raising an HTTPException would be wrong."
    Left uncommented at the Telegram provider bundle, per the finding's
    "at :2751" - the bundle's other members read the same way and one comment
    covers the pattern.

Process signal: `_run_scheduled_checks`' digest `health()` (`scufris/app.py:2751`)
was moved onto the service beyond the enumerated Steps. TASK.md's Close-out
justifies it, but DECISION.md enumerates the delegating surfaces exhaustively and
omits this fourth consumer.

- Answered: DECISION.md's surface table gains the digest `health()` row plus a
  sentence on how it was found (removing the last `agent_health` import) and why
  it could not be left behind. The enumeration is exhaustive again.

Process signal: `scufris/health.py:258` still reads a CODEX session count for a
claude or opencode orchestrator, on both the legacy and the scoped health
surface. Consistent between the two, so the contract test passes and it is not a
regression - but the epic's "stop leaking codex data" property is not complete
for the health surface. Close-out flags it; it wants a follow-up task, not a
finding here.

- Answered: seeded as 20260803-032950 ("Make the health session count follow the
  orchestrator backend") under the same epic, created in this worktree. Not
  fixed here.

Verification, independently re-derived in the worktree by the recording pass:

- `ruff check .` + `ruff format --check .` (194 files) + `mypy .` (194 files) -
  clean.
- `python -m pytest` - 982 passed, exit 0. Matches the Close-out number.
- `cd web && npm run ci` - exit 0.
- DoD proof `rg -n "resolve_codex_home" scufris/app.py` - empty, exit 1.
- Red-before-fix, by checking master's `scufris/app.py` into the worktree and
  re-running: `test_orchestrator_surfaces_are_backend_consistent` fails on all
  four backends and `test_legacy_agent_routes_delegate_to_scoped_diagnostics`
  fails. Both new tests are pinned at their own boundary.
- `scufris/agent_store/reserved.py:44` - the orchestrator record is synthetic and
  its `model` is `default_model_for(settings, backend)`, so `_require_agent`
  cannot 404 and the `/api/agent/info` model fix is real.
- `_agent_config` and the three moved sync routes are `def`, not `async def`, so
  the new store reads run on the threadpool and do not take SQLite's lock on the
  loop thread.
- R1.1's load-bearing claim re-derived directly:
  `web/src/agent-view.test.ts:104` returns `{}` for `/api/agent/usage`.
- `tatr check` - clean for this task.
- No `manual:` proofs in this DoD; nothing pending on that axis.

Not run: `nix flake check` and the KVM release tests.

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

Round 1's five findings are all confirmed fixed and ticked. R1.1's fix was
re-derived by reverting `renderUsage(quota.value)` to a raw-envelope pass: the
supported case goes red (`meter.hidden` true, expected false). R1.2's two
`codex_home` pins, R1.4's docstring and R1.5's comment are present as described.
R1.3's literal ask is met; what it does not yet pin is carried as R2.1.

- [ ] R2.1 (MINOR) tests/test_app.py:1855 -
  `test_disabled_agent_is_supported_not_unsupported` answers R1.3's ask
  (`enabled: false` beside a `supported: true` quota) but still cannot go red on
  DECISION-4. Its `codex_home` is EMPTY, so the deleted `settings.agent_enabled`
  short-circuit and the delegated reader agree exactly - `{"supported": true,
  "value": null}` and `session_count: 0` either way (re-derived against
  `git show master:scufris/app.py:3571-3603`). Every other new test pins
  `agent_enabled=True`, so restoring the short-circuit fails nothing. Point this
  test at a POPULATED home (`_write_session_rollout(...)` as at
  tests/test_app.py:1841) with `agent_enabled=False` and assert usage and memory
  report the real reading beside `enabled: false`.
  - Response:

- [ ] R2.2 (NIT) web/src/agent-view.test.ts:386 - "hides the meter when the
  backend cannot report usage" passes with or without the unwrap:
  `renderUsage` (`web/src/chat-sidebar.ts:165`) reads `usage?.primary`, which a
  raw `{supported: false, value: null}` also lacks, so the meter is hidden
  either way. Only the supported case is load-bearing. Assert something the
  envelope discriminates - e.g. that the meter is empty (`textContent` blank)
  as well as hidden - or drop the case.
  - Response:

Correcting the Round 1 record rather than rewriting it: Round 1's verification
line above reads "`python -m pytest` - 982 passed, exit 0. Matches the Close-out
number". The suite collects and runs 981 (`pytest --collect-only` counts 981;
two clean full runs, exit 0), and TASK.md's Close-out says 981. 982 was the
count before R1.3's deletion; the round-1 prose was not updated with it. The
Close-out is the correct record.

Verification, independently re-derived in the worktree by the recording pass:

- `ruff check .` clean; `ruff format --check .` 194 files; `mypy .` no issues in
  194 files - all exit 0.
- `python -m pytest` - exit 0 on two consecutive clean runs, 981 tests.
- `cd web && npm run ci` - exit 0 (260 tests, webpack build ok).
- R2.1's load-bearing claim re-derived from master's `get_usage`, `get_memory`
  and `get_account`: under an empty home the short-circuit and the reader are
  observationally identical.
- R2.2's load-bearing claim re-derived from `renderUsage`'s
  `if (!usage || !primary)` guard.
- No regression traced to any round-1 fix. `web/src/agent-view.ts` is untouched
  by 462c311, and the `codex_home` pins do not change what the health test
  asserts.

One flake observed, not a finding: `test_orchestrator_chat_uses_server_cwd`
(tests/test_app.py:2373) failed once under concurrent load and passed alone and
on every subsequent full run. It is a streaming/`_wait_state` timing test,
unmodified by this diff.

Not run: `nix flake check` and the KVM release tests.
