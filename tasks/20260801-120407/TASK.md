# Import legacy JSON state into the database

- STATUS: CLOSED
- PRIORITY: 81
- TAGS: bug, v0.2.0, reliability, storage, backend
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-120404

## Story

As a Scufris operator upgrading an existing install, I want my legacy JSON
state read into the database under a policy that backs up, validates and
refuses damage, so that the store cutover is a wiring change rather than a
data-loss risk.

## Steps

- [x] Write the failing proofs first, calling the import directly: a second run
      that is a no-op, a damaged file refused by name and location, and an
      invalid record that rolls the whole import back.
- [x] Add `scufris/db/legacy.py` with one importer for `projects.json` and the
      shared shape the two follow-up tasks reuse for their own stores.
- [x] Add a `legacy_import` table in its own Alembic revision, one row per
      completed source file, and gate each importer on it. This replaces
      DECISION.md section 4's `PRAGMA user_version` gate: the import needs
      `Settings.state_dir` and pydantic validation, neither of which belongs
      inside a schema revision.
- [x] Implement DECISION.md section 4's policy table verbatim: copy the source
      to `<name>.pre-sqlite.bak` before reading it; validate every record
      through its pydantic model (`Project`); refuse a damaged file with its
      path, line, column and message; NEVER delete a legacy file; run the whole
      import for one source inside one `db.transaction()`.
- [x] Make a validation failure fail the import rather than dropping the
      record. `ProjectStore._load` currently logs and skips an invalid record
      (`scufris/projects.py:107`); that tolerance is what the epic is removing,
      so do not port it.
- [x] Do NOT wire the import into startup here. It stays a tested, callable
      unit; the fourth task calls it in the same change that makes the database
      authoritative. Wiring it earlier would let a project created after the
      import land in `projects.json` and be lost at cutover.
- [x] Document the operator-facing consequences in `README.md`: the `.bak`
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
- All Python checks pass (cmd: `ruff check . && mypy . && python -m pytest`),
  excepting the two `tatr`-shelling tests that fail identically on the base
  branch and are owned by 20260802-191034.

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

## Close-out

### What and why

`scufris/db/legacy.py` is one function every store migration reuses -
`import_legacy_file(db, source, load)` - plus the one loader this task owns,
reached as `import_projects(db, state_dir)`. The whole of DECISION.md section
4's per-store policy is in the shared function rather than in each loader: the
source is copied to `<name>.pre-sqlite.bak` before it is read, it is never
deleted, a file that does not parse is refused with its path, line, column and
the parser's own message, and the import plus its `legacy_import` row are one
`db.transaction()`. A loader's only job is to validate its own records through
their pydantic model and write them on the connection it is handed.

Two choices are load-bearing and each has a proof that fails without it:

- **The `legacy_import` row is written inside the same transaction as the
  records it stands for.** It is the gate, so it can only exist if that import
  committed in full. A failure therefore leaves no rows AND no gate: the
  operator repairs the file and the retry starts from the beginning, rather
  than finding a store marked done and half-imported.
- **Records are inserted one at a time as they validate.** Validating the whole
  file up front would satisfy "no rows after an invalid record" without the
  transaction doing anything, so the rollback proof would pass with the
  transaction removed. Inserting as we go makes the earlier record's
  disappearance the thing being proven.

The gate is read INSIDE the transaction, so the check and the write cannot be
separated by a second process. That holds the write lock across a small file
read; it is affordable because this runs once, at startup, over files a single
host wrote by hand, and it is what keeps the whole function one atomic step.

`ProjectStore._load`'s tolerance is deliberately not ported: an invalid record
fails the import instead of being logged and skipped. Nothing under `scufris/`
outside the tests calls the importer - checked by grep, and the scope fence the
plan asked for.

The revision (`9b6587dab793`) was written through the documented maintainer
loop: scratch database, autogenerate, `ruff check --fix`, `ruff format`.

### Alternatives considered

- **A `LegacySource` dataclass registry (filename + loader).** One caller and
  one source today; the follow-up migrations can add a registry when they have
  more than one thing to register. The reusable shape is the `Loader` signature,
  which is what they actually need to agree on.
- **`record_count` on `legacy_import`.** Nothing reads it. The count goes to the
  log line instead; the row records the fact the gate needs and the timestamp an
  operator accounting for a migration would want.
- **Backing up with `shutil.copy2`.** It creates the target under the umask and
  copies the mode afterwards, leaving a window where a copy of - eventually -
  the auth session file is world-readable. The copy is created 0600 with
  `O_CREAT|O_EXCL|O_NOFOLLOW`, matching `backup_database`'s umask reasoning.
- **Reading the file before opening the transaction.** It would shorten the lock
  hold, but the gate check would then sit outside the transaction that acts on
  it, which is the check-then-act this table exists to make atomic.
- **Refusing a duplicate id in the legacy JSON with a message of its own.** Not
  added: the primary key already refuses it, loudly and inside the transaction,
  so the import is still all-or-nothing. A nicer message is worth writing when a
  real file produces one.

### Difficulties and diagnosis

The three named proofs passed on their FIRST run - the previous task's lesson
was to sabotage before calling a proof done, and that is what happened here, so
the red came from the sabotages rather than from an incomplete implementation.
Five sabotages, five intended failures (table below). The one that mattered was
the transaction sabotage: it confirmed the design note above, that up-front
validation would have made the rollback proof vacuous.

`test_declared_tables_are_the_only_ones` failed on the new revision, which is
the assertion doing its job - it exists to catch a table this epic did not
intend. Updated with a note that `legacy_import` is bookkeeping, not a store.

The predecessor's reflection asked for the pre-migration `VACUUM INTO` backup to
be proven on the real path at the next revision, because a one-revision history
cannot express "behind head". This task is that revision, so
`test_the_backup_is_taken_on_the_real_migration_path` now migrates a database to
the previous revision, gives it a row, upgrades it the way startup does, and
asserts the copy holds the row, the OLD revision and none of the new revision's
tables. The seam test's docstring no longer claims the branch is unreachable.

### Evidence

| Sabotage | What broke |
|-|-|
| `_is_imported` always False | `test_legacy_projects_import_is_idempotent_and_refuses_damage` (four rows on the second run) |
| invalid record logged and skipped, as `ProjectStore._load` does | `test_legacy_import_rolls_back_on_an_invalid_record` |
| `conn.commit()` after each insert (not one transaction) | the rollback proof, plus idempotency and the backup proof |
| the backup call replaced by its path | `test_legacy_import_backs_up_and_never_deletes_the_source` and the symlink proof |
| a damaged file parsed as `[]` | the damage half of proof 1 |
| the pre-migration `backup_database` call removed | `test_the_backup_is_taken_on_the_real_migration_path` |

- `ruff check .` clean, `mypy .` clean on 181 source files.
- `python -m pytest`: 932 passed, 2 failed - `test_project_tasks_endpoint` and
  `test_read_project_tasks_parses_real_tatr`, which fail identically on `master`
  at `e816f46` (both shell out to the real `tatr`, whose output moved).
  Pre-existing and unrelated; skipped under `nix flake check`, which is the
  canonical gate.
- `nix flake check`: all checks passed (ruff, mypy, pytest, filesize, records).
- Scope fence: `grep -rn "import_projects\|import_legacy_file" scufris/` finds
  only the definition and the `scufris.db` re-export. No call site.

### Reflection

Writing the sabotage as part of writing the proof - not after verification -
worked a second time, and this time it changed the IMPLEMENTATION rather than
the test: the transaction sabotage is what turned "validate everything, then
insert" into "insert as each record validates". A proof that cannot fail is a
design smell in the code as often as in the test.

The operator-facing README section is the first place the state directory is
documented as a thing an operator backs up and rolls back, rather than as an
environment variable. The `-wal`/`-shm` warning applies to the database that
landed last task, not to anything new here; it had no home until there was a
section about backups.

### Review round 1

Four findings, all answered (`REVIEW.md`). The one that mattered, R1.1, is a
kind of defect the sabotage habit does not catch: every proof was green and
stayed green, because the refusal MESSAGE was wrong rather than the behaviour.
It told the operator to restore `<name>.pre-sqlite.bak`, which on a refusal is
always a byte-identical copy of the damaged file this same run just made -
a no-op at best, and an overwrite of their own repair at worst. The wording came
from DECISION.md section 4's remedy, written before the backup was known to be
taken unconditionally before the parse; the implementation copied it verbatim,
which is what a "verbatim" instruction earns. The remedy is now the two that
work, pinned by assertions rather than left to prose, and the README bullet
says the same thing in operator words.

R1.2 narrowed the symlinked-backup refusal to `LegacyImportRefused`; the bare
`RuntimeError` was copied from `migrate.py`, where nothing catches a narrower
type. R1.4 - the DoD's `python -m pytest` being red on the base branch - is
20260802-191034 rather than a fix here.
