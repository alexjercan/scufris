# Review: Migrate auth, host, schedule, and digest state with a legacy JSON import path

- TASK: 20260801-100413
- BRANCH: fix/db-post-host-state

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) scufris/db/legacy/__init__.py:104 - `import_legacy_state`
  documents a repair-and-retry path that does not work, and the whole-directory
  import is this task's headline deliverable. The docstring claims a refusal
  "degrades correctly - with no mappings imported, migrating the record's own id
  IS the right answer", but the retry AFTER the repair is what breaks. With a
  damaged `sessions.json` beside an `agents.json` whose record carries a
  pre-registry `session_id` for the same agent, the first start refuses
  `sessions.json`, still imports `agents.json`, and `load_agents`
  (`loaders.py:127`) writes that agent's `agent_session` row. The gate row for
  `agents.json` is now committed. The operator repairs `sessions.json` as the
  refusal instructs and restarts; `load_sessions` (`loaders.py:171`) does a bare
  `insert(AgentSessionRow)` for the same `agent_id` and raises
  `sqlalchemy.exc.IntegrityError: UNIQUE constraint failed:
  agent_session.agent_id`. Nothing catches it - `import_legacy_file` catches
  nothing and `import_legacy_state` catches only `LegacyImportRefused` - so it
  leaves `open_state_database` as a raw driver error, and it is PERMANENT: the
  agents gate row means the conflicting write is never replayed or undone, so
  every subsequent start fails the same way. That is the Story's "upgrading never
  loses a login" inverted into "upgrading cannot boot". Make `load_sessions`
  upsert on `agent_id` (SQLite `on_conflict_do_update`, as `scheduler._write`
  already does) so the repaired mapping wins over the row the agent loader
  migrated, and add a test for the refuse-repair-restart sequence. Reproduced:
  fixture with `agents.json` = one agent `builder` carrying
  `"session_id": "stale-pre-registry"`, `sessions.json` written first as
  `'{"builder": '` then repaired to a valid `builder` mapping; the second
  `open_state_database` raises the IntegrityError above.
  - Response: fixed. `load_sessions` now upserts `agent_session` on `agent_id`
    (SQLite `on_conflict_do_update`) and deletes the agent's history rows before
    writing the ones the file declares, so the repaired file REPLACES what
    `load_agents` migrated rather than colliding with it. Replacing is also the
    correct rule and not just the one that avoids the error: `sessions.json` is
    the switcher's own record, and the id on an agent record was only ever the
    pre-registry stand-in, so merging would leave a stale id in the operator's
    chat list. The false claim in `import_legacy_state`'s docstring is rewritten
    to state the repair property and why it holds. Proof:
    `test_repairing_a_refused_sessions_file_completes_the_import` in
    `tests/test_db_legacy.py`, which reproduces the reviewer's fixture - it fails
    on the parent commit with exactly the reported
    `UNIQUE constraint failed: agent_session.agent_id` - and asserts the repaired
    mapping, its parent link and a history of `["builder-1"]` with the stale id
    gone. Recorded as DECISION.md 6, including that the defect predates this
    branch and why it is fixed here rather than deferred.

- [x] R1.2 (MINOR) tests/test_db_state_boundary.py:167 -
  `test_post_host_state_uses_declared_persistence_boundary` enumerates the six
  stores in a hand-written dict, so it proves the boundary for the stores that
  exist today and cannot fail for a store added tomorrow. Step 1 asked for
  "every app-owned store constructor in `create_app`". Derive the set instead -
  iterate `vars(app.state)` and assert every value whose type is defined under
  `scufris.` and that carries a `_db` attribute holds `app.state.db`, or assert
  the dict's keys against the store attributes `create_app` actually sets - so a
  seventh store has to opt in rather than be silently unchecked. The stray-JSON
  half of the test already generalises; only the first half does not.
  - Response: fixed, and it found a live gap rather than only a durability one.
    `_discover_stores` walks `app.state._state` (Starlette keeps its attributes
    there, not in `__dict__`) plus one level of nesting for
    `HostScheduler.store`, and collects every `scufris`-defined object whose
    class name ends in `Store`. The six names are kept as an asserted FLOOR, so a
    walk that silently found nothing fails instead of passing vacuously.
    Discovery immediately surfaced a seventh store the list had missed:
    `ConfigChangeStore` (`scufris/hostconfig/changes.py`) is still an in-memory
    bounded `OrderedDict`, so this branch's "every app-owned store" claim was not
    true as written. Migrating a fifth store is materially outside this task's
    Steps, so it is now task 20260803-002141 under the same epic, and the test
    excludes `config_changes` BY NAME against that ID - with an assertion that
    the exclusion is still needed, so the task's own proof is deleting it.
    DECISION.md's consequence about the boundary is corrected to say the claim
    holds with one declared exception.

Verified independently, not taken from the branch's own claims:

- `ruff check .`, `ruff format --check .` (191 files) and `mypy .` (191 files)
  all clean; full `pytest` exits 0.
- All seven named DoD tests pass when run as a set:
  `test_concurrent_state_mutations_survive_restart`,
  `test_post_host_state_migrates_transactionally`,
  `test_host_proposal_decisions_survive_restart`,
  `test_privileged_audit_remains_an_external_boundary`,
  `test_legacy_json_state_migrates_idempotently`,
  `test_post_host_state_uses_declared_persistence_boundary`,
  `test_schema_has_no_pending_autogenerate_diff`.
- DoD greps: `! rg 'with_suffix\("\.json\.tmp"\)' scufris/` and
  `! rg "are still JSON" README.md` both hold; the doc grep matches in both
  READMEs.
- `python examples/state_migration.py` exits 0.
- Re-derived the load-bearing `_decide` atomicity claim rather than trusting
  `tests/test_host_action_decisions.py`: 40 threads released from one barrier
  against a single proposal produced exactly 1 approval and 39 `AlreadyDecided`,
  with the surviving row's `decided_by` equal to the winner. The
  read-check-write-in-one-immediate-transaction claim in `host_actions.py` holds.
- Re-derived the `seq = max(seq) + 1` claim, which the branch asserts is safe
  because the begin is immediate: 30 concurrent `put`s produced 30 rows and no
  `UNIQUE constraint failed: host_action.seq`.
- Audited every call site the offload work depends on. `rg` for
  `sessions\.(get|create|revoke|prune)` and for the `HostApprovalService`
  methods that became `async def`: every event-loop caller is offloaded or
  awaited, and the one un-offloaded `sessions.prune` (app.py:1107) is in
  synchronous `create_app`, before a loop exists, as its comment states.
- Read the Alembic revision against `models.py`; the four tables match the
  Step 2 spec column for column, including `host_action.seq` unique and
  `digest.id` autoincrement.
- `scufris/checks.py` is reformatting only, unrelated to this task. Harmless,
  and not counted as a finding.

Process signal: R1.1's underlying defect predates this branch - master's
`import_agent_state` has the same collect-refusals loop, the same
sessions-before-agents ordering and the same "degrades correctly" sentence. It is
raised here rather than deferred because this diff rewrote that docstring, moved
it to the new single entry point, and widened the policy from two per-half entry
points to one whole-directory loop, which is the change the Close-out argues for.
If the fix is preferred as its own task, the branch would still need the false
claim removed from `import_legacy_state`'s docstring before it lands.

## Round 2

- REVIEWER: in-session (this session's operating rules forbid spawning a
  subagent, so the round-1 reviewer also implemented the fixes; the compensating
  check is that each fix was verified by reverting it and watching the proof go
  red, not by re-reading the diff)
- VERDICT: APPROVE

Both round-1 findings are fixed and verified. Nothing new.

- R1.1 verified. `load_sessions` upserts `agent_session` on `agent_id` and
  deletes the agent's history rows before writing the file's own
  (`loaders.py:170-200`). Checked the fix is load-bearing rather than
  coincidental: restored the parent commit's `loaders.py` under the new test and
  it failed with exactly the reported
  `UNIQUE constraint failed: agent_session.agent_id` on
  `('builder', 'codex', 'builder-1', 'orchestrator', 'orch-1')`; restored the fix
  and it passes. The test asserts more than the absence of a crash - the repaired
  mapping wins (`builder-1`), its parent link survives, and the history is
  `["builder-1"]` with the stale pre-registry id gone rather than left beside it.
  The rewritten `import_legacy_state` docstring now states the repair property
  and why it holds instead of the claim that was false.
- R1.2 verified. `_discover_stores` walks `app.state._state` plus one nesting
  level and selects `scufris`-defined classes whose name ends in `Store`; the six
  known names are asserted as a floor, so a walk that found nothing fails rather
  than passing vacuously. Confirmed the walk actually discriminates: it found
  `config_changes`, which the hand-written list had missed and which does not
  hold the `Database`. That is now 20260803-002141, excluded by name, with
  `assert name in stores` so the exclusion cannot outlive the migration it waits
  on. The task's boundary claim is corrected in DECISION.md rather than left
  overstated.

Re-ran the whole verification, not just the tests near the fixes:
`ruff check .`, `ruff format --check .` (191 files) and `mypy .` (191 files)
clean; the full suite exits 0; all six named DoD tests plus
`test_repairing_a_refused_sessions_file_completes_the_import` pass; the three
DoD greps hold; `python examples/state_migration.py` exits 0;
`tatr check` clean on both 20260801-100413 and 20260803-002141.

Process signal: both findings had one root - a record asserting a property that
nothing executed. "Degrades correctly" described a recovery path no test walked,
and "every app-owned store" quantified over a set a hand-written list could not
falsify. Each was false, and each became checkable only when something ran it.
Worth carrying into the epic's remaining store cutovers: a docstring that claims
a recovery path owes a test that walks it, and one that quantifies over a set
owes a test that derives the set.
