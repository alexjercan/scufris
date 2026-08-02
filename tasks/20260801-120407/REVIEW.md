# Review: Import legacy JSON state into the database

- TASK: 20260801-120407
- BRANCH: fix/legacy-json-import

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

Verified independently: `ruff check .` clean; `mypy .` clean (181 files);
`tests/test_db_legacy.py` + `tests/test_db_migrations.py` 21 passed; the two
`python -m pytest` failures (`test_project_tasks_endpoint`,
`test_read_project_tasks_parses_real_tatr`) reproduce identically on `master`
at `e816f46`, so the close-out's pre-existing claim holds; the scope fence
grep finds no call site under `scufris/`; `tatr check` exits 0. The atomicity
claim was re-derived from `scufris/db/engine.py:239` - the begin really is
`BEGIN IMMEDIATE`, so the gate read and the `legacy_import` write cannot be
separated by a second process. R1.1 was re-derived by running the import
against a damaged file and diffing the bytes it leaves behind.

- [x] R1.1 (MAJOR) scufris/db/legacy.py:157 - the refusal message tells the
  operator to "Restore `<name>.pre-sqlite.bak`", but `_back_up` runs BEFORE
  `_parse` and unlinks any earlier copy, so on this path the backup is always a
  byte-identical copy of the damaged file this same run just made. Measured:
  `backup == damaged source: True`, which
  `test_legacy_projects_import_is_idempotent_and_refuses_damage`
  (`tests/test_db_legacy.py:129`) also pins. Following the advice is a no-op at
  best, and overwrites the operator's own repair with the damaged bytes at
  worst. Drop the restore clause and name the remedies that work: repair the
  file from the operator's own backup, or move it aside to import the rest of
  the state without it.
  - Response: fixed in this round's commit. The message now says the `.bak` is
    a copy of the same damaged file and that restoring it changes nothing, and
    names the two remedies that work. `scufris/db/legacy.py:156`. Pinned by two
    assertions in the damage half of
    `test_legacy_projects_import_is_idempotent_and_refuses_damage`; sabotaged
    back to the old wording, that test fails.
- [x] R1.2 (MINOR) scufris/db/legacy.py:134 - a symlinked backup target is a
  legacy file that cannot be trusted, but it raises a bare `RuntimeError`
  rather than `LegacyImportRefused`. The cutover's call site will catch the
  documented exception and let this one through as an unhandled error. Raise
  `LegacyImportRefused` and widen `test_a_symlinked_backup_target_is_refused`
  (`tests/test_db_legacy.py:170`) to assert the type.
  - Response: fixed in this round's commit. `scufris/db/legacy.py:133` raises
    `LegacyImportRefused` with the `REFUSED:` prefix the other refusals use,
    and `test_a_symlinked_backup_target_is_refused` asserts that type.
    `migrate.py:148` keeps its bare `RuntimeError` - it guards the database's
    own backup, which has no legacy-import caller to catch a narrower type.
- [x] R1.3 (MINOR) README.md:162 - "the startup fails rather than presenting
  you with an empty store" states as current behaviour something no code path
  performs: nothing calls the importer this release, which the CHANGELOG entry
  says explicitly. Match the surrounding bullet's conditional voice, e.g. "as
  each store moves, a file that does not parse is refused by name ... and the
  startup fails rather than ...".
  - Response: fixed in this round's commit. `README.md:161` now reads "When a
    store does move, a JSON file that does not parse is refused by name ...",
    and the same bullet states that a `.bak` is a copy rather than a repair -
    the operator-facing half of R1.1.
- [x] R1.4 (MINOR) tasks/20260801-120407/TASK.md:58 - the DoD's `cmd:` proof
  `ruff check . && mypy . && python -m pytest` is red on this branch (2
  failures) and red on `master`, so the criterion as written cannot be met
  here. Not this task's defect, and the close-out discloses it accurately. File
  a task for the two `tatr`-shelling tests that drifted, and either point this
  DoD at `nix flake check` - the gate the close-out calls canonical - or leave
  the line and reference the new task from it.
  - Response: filed as 20260802-191034 (repair the two `tatr`-shelling project
    task tests), created in this worktree. The DoD line keeps its command and
    now names that task as the owner of the exception. Not fixed here: the
    drift is in `tatr`'s `ls` output, which this branch does not touch.

- Process signal: the three named proofs passed on their first run and the red
  came from deliberate sabotages instead. That is the practice the previous
  task's reflection asked for, and it worked - the transaction sabotage changed
  the implementation, not just the test. Worth carrying as a plan step rather
  than a habit that depends on the implementer remembering it.

Pending user checks: none. This task's DoD has no `manual:` proof; the epic's
`manual:` item belongs to the epic, not to this task.

Inspection commands:

```sh
cd "$(sprout show fix/legacy-json-import)"
ruff check . && mypy .
python -m pytest tests/test_db_legacy.py tests/test_db_migrations.py
grep -rn "import_projects\|import_legacy_file" scufris/
```

## Round 2

- REVIEWER: in-session (subagent delegation is off for this session; the round
  verifies three small fixes and one filed task, each re-derived by running the
  code rather than reading the diff)
- VERDICT: APPROVE

All four round-1 findings verified fixed and ticked above:

- R1.1: the refusal against a damaged file now reads "... `<path>.pre-sqlite.bak`
  is a copy of this same damaged file, not a repair - restoring it changes
  nothing. Repair the file from your own backup, or move it aside ...". Ran the
  import against a damaged `projects.json` to read the real message.
- R1.2: a symlinked backup target raises `LegacyImportRefused`, confirmed by
  running it (`LegacyImportRefused REFUSED: ... is a symlink`). `migrate.py:148`
  keeping its bare `RuntimeError` is accepted - nothing catches a narrower type
  there.
- R1.3: `README.md:162` now scopes the claim with "When a store does move", and
  the bullet states a `.bak` is a copy rather than a repair.
- R1.4: 20260802-191034 exists in this worktree with the two failing tests as
  its DoD; the DoD line here names it as the owner of the exception.

No regressions: `nix flake check` passes (ruff, mypy, pytest, filesize,
records); `python -m pytest` is 932 passed with only the two pre-existing
`tatr`-shelling failures; `tatr check` exits 0. Sabotaging each fix back to its
round-1 form fails exactly the test that pins it, so neither guard is vacuous.

No new findings.

Pending user checks: none.
