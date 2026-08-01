# Import legacy JSON state into the database

- STATUS: OPEN
- PRIORITY: 81
- TAGS: bug, v0.2.0, reliability, storage, backend
- KIND: TASK
- FLOW STEP: PLANNED
- PLAN STATUS: APPROVED
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-120404

## Story

As a Scufris operator upgrading an existing install, I want my legacy JSON
state read into the database under a policy that backs up, validates and
refuses damage, so that the store cutover is a wiring change rather than a
data-loss risk.

## Steps

- [ ] Write the failing proofs first, calling the import directly: a second run
      that is a no-op, a damaged file refused by name and location, and an
      invalid record that rolls the whole import back.
- [ ] Add `scufris/db/legacy.py` with one importer for `projects.json` and the
      shared shape the two follow-up tasks reuse for their own stores.
- [ ] Add a `legacy_import` table in its own Alembic revision, one row per
      completed source file, and gate each importer on it. This replaces
      DECISION.md section 4's `PRAGMA user_version` gate: the import needs
      `Settings.state_dir` and pydantic validation, neither of which belongs
      inside a schema revision.
- [ ] Implement DECISION.md section 4's policy table verbatim: copy the source
      to `<name>.pre-sqlite.bak` before reading it; validate every record
      through its pydantic model (`Project`); refuse a damaged file with its
      path, line, column and message; NEVER delete a legacy file; run the whole
      import for one source inside one `db.transaction()`.
- [ ] Make a validation failure fail the import rather than dropping the
      record. `ProjectStore._load` currently logs and skips an invalid record
      (`scufris/projects.py:107`); that tolerance is what the epic is removing,
      so do not port it.
- [ ] Do NOT wire the import into startup here. It stays a tested, callable
      unit; the fourth task calls it in the same change that makes the database
      authoritative. Wiring it earlier would let a project created after the
      import land in `projects.json` and be lost at cutover.
- [ ] Document the operator-facing consequences in `README.md`: the `.bak`
      files, that legacy JSON is never deleted, that `-wal`/`-shm` belong in any
      backup, and that downgrade works only while the legacy files still exist
      and is one-way once the operator deletes them - in those words.

## Definition of Done

- A legacy `projects.json` imports once, a second run is a no-op, and a damaged
  file is refused with its location named
  (test: `test_legacy_projects_import_is_idempotent_and_refuses_damage`).
- The source file is backed up and never deleted
  (test: `test_legacy_import_backs_up_and_never_deletes_the_source`).
- One invalid record fails the whole import and leaves no rows behind
  (test: `test_legacy_import_rolls_back_on_an_invalid_record`).
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`).

## Notes

- Epic: 20260729-102145. Lane B, third of four. Depends on the Alembic runner.
- Ordering rationale: the import lands BEFORE the cutover, not after. The
  alternative order strands an operator's projects for a whole task, which the
  original single-task plan flagged and could not fix within one lane.
- Scope fence for review, not a DoD proof (it is green on the base branch and
  so proves nothing): nothing under `scufris/` outside the tests calls the
  importer when this task lands. The fourth task adds the only call site.
- The two follow-up store migrations (20260801-100409, 20260801-100413) add
  their own importers and `legacy_import` rows against this shape.
