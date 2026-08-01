# Review: Spike: choose the persistence mechanism, migration, and recovery policy

- TASK: 20260801-100405
- BRANCH: master (landed 6ca5240, 8948247; no sprout - the deliverable is two
  records plus a measurement harness, and no `scufris/` or `tests/` file is
  touched)

## Round 1

- REVIEWER: in-session (this session's operator rules prohibit subagent
  delegation, so the out-of-context default could not be used; compensated by
  re-deriving every cited code location from the source, re-running the harness,
  and independently counting the test suite - see Verification below)
- VERDICT: REQUEST_CHANGES

- [ ] R1.1 (MAJOR) tasks/20260801-100405/DECISION.md:187 - the migration policy
  contradicts the lane it is written for. Section 4 says the import is "One
  entry point migrates the WHOLE state directory as a single transaction - not
  one transaction per store" and that "Partial migration is not a state that can
  exist". The three implementation tasks import in three phases:
  20260729-102147 imports `projects.json` alone at `user_version=1` (a Step this
  spike itself added), 20260801-100409 imports agent/session/outcome/settings/
  reasoning, and 20260801-100413 imports auth/host/schedule/digest. Between
  those tasks the state directory IS partially migrated, by design and for good
  reason - each task has to land green on its own. An implementer following
  DECISION.md literally at 20260729-102147 must either import every store (which
  breaks that task's deliberate pilot-only scope) or violate the decision.
  Change: state the invariant at the level it actually holds - each
  `user_version` step is atomic for the stores it covers, the directory import
  completes across versions 1..N, and a store not yet at its version keeps
  reading its legacy JSON. Then say what "partial" is being ruled out: a store
  half-imported, not a directory half-migrated.

- [ ] R1.2 (MAJOR) tasks/20260801-100405/SPIKE.md:186 - a load-bearing number
  no committed artifact reproduces. The isolation section quotes "6.6 / 6.2 /
  3.2ms for FULL / NORMAL / OFF" from "an isolated seven-table measurement on an
  otherwise idle machine" and uses it to argue the in-scenario 35 / 16 / 6ms is
  inflated. That measurement was an ad-hoc heredoc, not `bench_persistence.py`,
  and it used a SINGLE-table schema with connections closed per iteration - not
  seven tables, as the prose claims. The committed `isolation` scenario prints
  only the higher figures. The spike rule is a citation or a reproducible
  artifact for every claim, and this is the number the "seconds, not minutes"
  conclusion rests on. Change: either fold the isolated variant into
  `scenario_isolation` so the quoted numbers come out of the committed harness,
  or delete the sentence and derive the conclusion from what the harness prints.

- [ ] R1.3 (MAJOR) tasks/20260801-100405/SPIKE.md:190 - the test-suite size is
  wrong and the conclusion drawn from it is not re-derived. "Across a ~600-test
  suite that is seconds, not minutes" understates it: `nix develop --command
  python -m pytest --collect-only` reports **896 tests collected**. At the
  in-scenario 28.963ms that is 26s, which is not "seconds" in the sense the
  sentence implies, and it is the endpoint a reader will check first because
  R1.2's cheaper number is the unreproducible one. Change: use the real count,
  give the range both endpoints produce, and note that only tests taking a store
  fixture pay it at all - which is the honest reason the cost is affordable.

- [ ] R1.4 (MINOR) tasks/20260801-100405/SPIKE.md:242 - the one sub-axis the
  rejected candidate wins is printed and then walked past. In the events block
  JSON retention costs 5.26ms against SQLite's 10.56ms; the surrounding prose
  covers append cost, pagination and size but never acknowledges that the
  retention delete goes the other way. A record that names 4x disk as a cost
  should name this too. Change: one sentence saying JSON's whole-file rewrite
  makes "drop all but the last 1000" a single serialization while SQLite deletes
  4000 rows through the WAL, and that this is the one measured axis favouring
  the rejected design.

- [ ] R1.5 (MINOR) tasks/20260801-100405/DECISION.md:225 - `scufris/checks.py`
  is named as the home for `PRAGMA integrity_check`, but that module is the HOST
  check registry: `check_disk`, `check_failed_units`, `check_thermal`,
  `check_store`, `check_flake`, all driven by `HostInspector`. Only
  `check_scufris(health: AgentHealth)` is app-facing, and it takes agent health,
  not a database handle. Constraining the implementer to that module by name is
  a decision the evidence does not support. Change: say the integrity check is
  exposed through the existing check surface without naming the module, or name
  it and say why a host-inspection registry is the right place for an app-state
  probe.

## Verification

Re-derived independently of the spike's own text:

- Every code location cited in SPIKE.md and DECISION.md was read at the line
  given: `scufris/app.py:3739` (`uvicorn.run`, no workers),
  `scufris/config.py:52` (`state_dir`), `scufris/mcp_server.py:81` (a second
  process constructing `AgentStore` and `ProjectStore`),
  `scufris/agent_store/store.py:502-506` (registry, outcome, agent row - three
  independent writes), `scufris/reasoning_store.py:82-86` (load-append-persist)
  and `:120` (the swallowed `OSError`), `scufris/scheduler.py:107` (`get` on a
  write path), `scufris/host_actions.py:182` (`HostActionStore`, in-memory),
  `scufris/host_approvals.py:287` (`refresh_pending`), `scufris/auth/store.py:75`
  (0600 via `os.open`, not chmod-after-write). All accurate.
- `bench_persistence.py` re-run per scenario; every quoted block is a verbatim
  line from a run in this session. `ruff check`, `ruff format --check` and
  `mypy` pass on it; the 12 pre-existing format deviations elsewhere in the repo
  are untouched by this diff.
- `sqlite3.threadsafety == 3`, `journal_mode=wal`, `busy_timeout` and
  `PRAGMA user_version` confirmed in the devshell (python 3.14.6, sqlite
  3.53.1) before being written into DECISION.md.
- Test count taken from `pytest --collect-only`, which is what produced R1.3.
- The four DoD command proofs pass: 31, 8, 2 and 8 matches respectively.

Not findings, recorded so a later round does not re-raise them:

- Scenario 2 injects an exception rather than killing the process. That models
  what 20260729-102146 actually observed (a `FileNotFoundError` raised partway
  through `mark_finished`), so it is the right injection; scenario 4 covers the
  kill separately.
- Thread-local connections accumulate across two pools (anyio's for sync routes,
  asyncio's default executor for `to_thread`). Both are bounded, so the
  connection count is bounded. No leak.
- DECISION.md requires the MCP subprocesses to share the database. Nothing needs
  assigning for it: they build `AgentStore`/`ProjectStore` from `Settings`
  (`mcp_server.py:81`), so they follow those stores onto the core automatically.

## Pending user checks

- (accepted 2026-08-01) The durability and migration tradeoffs of the selected
  architecture. Recorded in the epic's Manual Acceptance.
