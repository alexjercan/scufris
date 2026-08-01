# Review: Add the SQLAlchemy transactional engine core

- TASK: 20260729-102147
- BRANCH: fix/sqlalchemy-engine-core

## Round 1

- REVIEWER: out-of-context (three lanes: behavior/proofs, correctness/security/concurrency, design/standards/docs)
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) scufris/db/engine.py:103 - `transaction()` is not reentrant
  and nothing says so. A nested `db.transaction()` on the SAME thread checks out
  a second pooled connection, whose `BEGIN IMMEDIATE` blocks on the outer
  transaction's write lock for the full `busy_timeout` and then raises
  `OperationalError: database is locked`, which also rolls back the outer unit of
  work. Re-derived in-session: the nested call failed after 5.01s. This is the
  boundary three follow-up store tasks build on, and "store A's unit of work
  calls store B's" is the natural mistake; it fails slowly and reads as external
  contention rather than as a bug. Fail fast instead: track the active
  `Connection` in a `ContextVar`, raise `RuntimeError` naming the nesting on
  re-entry, and pin it with a test. Do NOT make re-entry silently reuse the
  connection - an inner `with` block that appears to commit but does not is a
  worse failure than the one being fixed.
  - Response: fixed. `transaction()` now guards on a module-level `ContextVar` and raises `RuntimeError` naming the nesting; the fail-fast option was taken, not silent reuse. Two tests: `test_nested_transactions_are_refused_immediately` (which asserts elapsed < 1.0s, so a guard that merely renamed the same 5s deadlock would fail it) and `test_the_nesting_guard_is_per_context_not_global`. Falsified: removing the guard fails the first with the real 5s `OperationalError`; swapping the ContextVar for a module-level flag fails the second plus both existing concurrency tests.
- [x] R1.2 (MINOR) scufris/db/engine.py:166 - `_secure(path)` runs OUTSIDE the
  `try/except` that disposes the engine, so a failing `os.chmod` (read-only
  mount, foreign uid, immutable flag) propagates out of `open_database` having
  leaked the engine and its already-dialed pooled connection. Move `_secure(path)`
  inside the same `try:` whose `except` calls `engine.dispose()`.
  - Response: fixed. `_secure(path)` moved inside the `try:` whose `except` calls `engine.dispose()`.
- [x] R1.3 (MINOR) scufris/db/engine.py:124 and scufris/README.md:383 - the claim
  that a corrupt file raises "at `open_database`" overstates what is implemented.
  Re-derived in-session: a database with an intact page-1 header and zeroed later
  pages opens CLEANLY and raises `DatabaseError: database disk image is malformed`
  only at the first query; `test_damaged_state_refuses_to_load` covers only a file
  that is not SQLite at all. The DoD's substance holds - damage is never presented
  as empty - but the wording does not. Either run `PRAGMA quick_check` inside
  `open_database`, or narrow both wordings to "damage is never presented as empty:
  it surfaces as `DatabaseError` at open or at first read".
  - Response: fixed by narrowing the wording, not by `quick_check` - a whole-file check at every startup costs more than the claim is worth, and the guarantee that matters (never empty) holds either way. Both `engine.py` and README section 9 now say open OR first read. Pinned by `test_corrupt_pages_behind_a_good_header_raise_at_the_first_read`, which builds a real 500-row database and zeroes every page after the header.
- [x] R1.4 (MINOR) scufris/db/engine.py:172 - the `SIDECAR_SUFFIXES` half of
  `_secure` never runs on the fresh-open path: re-derived in-session, the
  directory right after `open_database` contains only `scufris.db`, so both
  sibling chmods hit `FileNotFoundError`. `test_state_database_files_are_owner_only`
  therefore proves SQLite's mode inheritance, not this loop - reducing the loop to
  `for candidate in (path,):` leaves all 8 tests green. The loop does earn its
  place for leftover sidecars from a crashed run, so keep it and cover it: add a
  test that pre-creates `scufris.db-wal` and `scufris.db-shm` at 0644 before
  `open_database` and asserts 0600 after.
  - Response: fixed. The loop stays and its docstring now says which case it is for; `test_sidecars_left_behind_by_a_crash_are_narrowed_on_open` pre-creates both sidecars at 0644. Falsified: reducing the loop to `(path,)` fails that test.
- [x] R1.5 (MINOR) scufris/db/engine.py:60 - the comment
  `# DECISION.md section 1 (20260801-100405), unamended by the SQLAlchemy swap.`
  names a decision ID in a code comment, which `AGENTS.md:103` forbids outright
  ("Task IDs belong in task records and Markdown, never in code comments or
  docstrings"). Confirmed it is the only such reference left under
  `scufris/**/*.py`. Delete that line; the three lines after it already state the
  invariant as a fact about the code.
  - Response: fixed. The decision ID is gone; the three lines that state the invariant remain.
- [x] R1.6 (MINOR) scufris/db/__init__.py:24 - `FILE_MODE`, `PRAGMAS` and
  `SIDECAR_SUFFIXES` are re-exported in `__all__` with zero callers anywhere in
  `scufris/`, `tests/` or `examples/`, and README section 9 does not list them as
  public. Speculative surface against the repo's YAGNI rule. Drop the three from
  both the `__init__` import and `__all__`, leaving them constants in `engine.py`.
  - Response: fixed. `FILE_MODE`, `PRAGMAS` and `SIDECAR_SUFFIXES` dropped from the import and `__all__`; they stay in `engine.py`.
- [x] R1.7 (MINOR) scufris/README.md:366 - "The public surface is four names"
  undercounts the table below it, which lists five, and omits `Database.path` and
  `Database.close()` - both public and both used by the tests and the `database`
  fixture. This section exists so the follow-ups do not re-derive the surface, so
  it has to be right: add rows for `Database.path` and `Database.close()` and drop
  the numeral in favour of "the names below".
  - Response: fixed. Rows added for `Database.path` and `Database.close()`, and the count replaced with "the names below".
- [x] R1.8 (NIT) scufris/db/engine.py:67 - `PRAGMA busy_timeout=5000` is applied
  third, so `journal_mode=WAL` and `synchronous=FULL` run on a connection whose
  timeout this module has not set yet. Harmless today only because pysqlite's own
  `timeout=5.0` connect default already installs a busy handler. Move
  `busy_timeout` to the front of `PRAGMAS` so the guarantee does not rest on a
  driver default.
  - Response: fixed. `PRAGMA busy_timeout=5000` is now first, with a comment saying why.
- [x] R1.9 (NIT) scufris/db/engine.py:134 - `os.open` and `os.chmod` both follow
  symlinks, so if `state_dir/scufris.db` is a symlink, `open_database` initializes
  the database through it and `_secure` chmods the target. Only reachable if
  something untrusted can create names in the state dir, which is why this is a
  NIT and not more. Add `os.O_NOFOLLOW` to the create and refuse rather than
  chmod any candidate where `Path.is_symlink()`.
  - Response: fixed, and covered rather than just changed. `O_NOFOLLOW` on the create, plus `_secure` refuses any symlinked candidate - which also covers a symlinked `-wal`/`-shm`, where `O_NOFOLLOW` on the database path cannot help. `test_a_symlinked_database_path_is_refused` asserts the target keeps its 0644 mode; dropping `O_NOFOLLOW` fails it.
- [x] R1.10 (NIT) scufris/db/__init__.py:5 - `see [the module docstring](engine.py)`
  puts a Markdown link inside a Python docstring; no other package `__init__` in
  the repo does this. Replace with a plain reference to `scufris/db/engine.py`.
  - Response: fixed. Plain reference to `scufris/db/engine.py`.
- [x] R1.11 (NIT) scufris/db/engine.py:80 - the class docstring justifies the
  injected engine with "so tests and the migration runner can share one"; no
  migration runner exists yet and no test constructs `Database` directly. Cut the
  clause and keep "Construct through :func:`open_database`."
  - Response: fixed. Reduced to "Construct through :func:`open_database`."
- [x] R1.12 (NIT) scufris/db/engine.py:144 - `dbapi_connection: object` forces a
  `# type: ignore[attr-defined]` on the very next line. Annotate the parameter
  `sqlite3.Connection` (adding the stdlib import) and drop the ignore if mypy
  accepts the listener signature.
  - Response: fixed. Annotated `sqlite3.Connection`; mypy accepts the listener signature and the `type: ignore` is gone.
- [x] R1.13 (NIT) tests/conftest.py:146 - "The measured cost is ~10ms per test"
  quotes the persistence spike's rig, not this one, and reads as if measured here.
  Measured in-session on this rig: 4.92ms per open/close. Drop the number and keep
  the reason the fixture is file-backed.
  - Response: fixed. Replaced with "single-digit milliseconds", which is what this rig measures.

### What the primary re-derived

The in-session pass re-ran every proof and independently reproduced the
load-bearing claims rather than accepting them from the lanes:

- Nested `transaction()`: raises `OperationalError` after 5.01s (R1.1).
- Partial corruption: opens clean, raises `DatabaseError` at first read (R1.3).
- Sidecars after a fresh `open_database`: only `scufris.db` on disk (R1.4).
- Permissions under `umask 000`: `scufris.db`, `-wal` and `-shm` all 0600.
- A file that is not SQLite at all: raises `sqlalchemy.exc.DatabaseError` at open.
- Fixture cost on this rig: 4.92ms per open/close (R1.13).
- The AGENTS.md task-ID rule and the zero-caller status of the three re-exports.

### Checks

- `nix build .#scufris`: passes (the lanes skipped it; the primary ran it).
- `ruff check .` and `mypy .`: clean, 172 source files.
- `python -m pytest`: 902 passed, 2 failed. Both failures reproduce identically
  on master in the main checkout and are unrelated to this diff; filed as
  20260801-123345 rather than fixed here.
- `python scripts/check_file_size.py`: passes. `engine.py` 178 lines,
  `__init__.py` 30, `test_db_engine.py` 250, no ALLOWLIST change.
- `ruff format --check .`: 12 pre-existing unformatted files, none in this diff.

### Honesty

The close-out's claims hold. A lane independently rebuilt both sabotaged engines
outside the repository and confirmed the two pool proofs discriminate exactly as
recorded: with the begin hook removed the `BEGIN IMMEDIATE` probe is not refused,
and with the pragmas applied once at open `foreign_keys` reads `[1, 0]` across two
pooled connections against `[1, 1]` on the shipped hook. The recorded 902/2 test
split, the 172-file mypy count, the sqlalchemy 2.0.51 / greenlet 3.5.4 versions
and the documented `connect_args={"isolation_level": None}` deviation from Step 6
all check out.

### Pending user checks

None. This task has no `manual:` proof; the epic's manual criterion belongs to
20260729-102145 and cannot be exercised until a store actually moves.

- Process signal: the branch's `tasks/20260729-102147/TASK.md` still reads
  `STATUS: OPEN` / `FLOW STEP: PLANNED`, because `tatr flow` mutates the MAIN
  checkout's copy while the worktree carries its own. A squash-merge would
  regress the header. Reconcile at landing rather than on the branch.
- Process signal: the branch is one squashed commit, so Step 1's "write the
  failing proofs first" is not observable from history - only the close-out
  asserts it. The falsification evidence is what substitutes, and it is stronger.
- Process signal: three of the thirteen findings (R1.1, R1.3, R1.4) are gaps
  between what the documentation promises and what the boundary implements, on a
  task whose whole purpose is that the follow-ups can trust the boundary without
  re-deriving it. The plan asked for the API to be recorded; it did not ask for
  the recorded API to be tested against the code.

## Round 2

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

Every round-1 finding was verified CONFIRMED by an out-of-context reviewer that
did not run round 1: it read the code rather than the `Response:` lines, re-ran
all four recorded fix falsifications on a clone outside the worktree (guard
removed, ContextVar swapped for a module flag, sidecar loop reduced to
`(path,)`, `O_NOFOLLOW` dropped) and found no over- or under-claim, and
re-measured the R1.13 fixture cost at 2.45ms median. It also probed the guard's
exception and abandonment paths, a symlinked state dir, symlinked sidecars, and
engine disposal on the `_secure` raise, all clean. All thirteen boxes are ticked
on that confirmation.

Two regressions from the round-1 fixes themselves:

- [x] R2.1 (MINOR) scufris/db/engine.py:88 - the nesting guard was a module-level
  `ContextVar[bool]`, so it was global to the PROCESS rather than to a database.
  Two units of work on two different files - which cannot contend, and so are not
  the deadlock the guard exists to name - were refused with a message that
  misstates the cause. Store the active database's path in the ContextVar and
  refuse only when it equals this database's.
  - Response: fixed. `_open_transaction` is now `ContextVar[Path | None]` holding
    the open database's path, and the message names that path.
    `test_the_nesting_guard_is_per_database_not_per_process` opens two databases
    and nests one inside the other. Falsified: reverting the condition to
    `is not None` fails exactly that test.
- [x] R2.2 (NIT) scufris/db/engine.py:159 - the `O_NOFOLLOW` refusal surfaced as a
  bare `OSError: [Errno 40] ELOOP`, while the sibling path one function down got
  an explanatory `RuntimeError`. Wrap the `os.open` and re-raise with the
  `_secure` wording, and add the symlink refusal to README section 9's failure
  modes.
  - Response: fixed. The `os.open` is wrapped; a symlinked path raises
    `RuntimeError: ... is a symlink; refusing to open the database through it`,
    and any other `OSError` still propagates unchanged rather than being
    relabelled. README section 9 now lists the symlink rule, including that a
    symlinked state DIR is still fine. The test asserts the target keeps its 0644
    mode and is never written through.

### Checks after the round-2 fixes

- `ruff check .` and `mypy .`: clean, 172 source files.
- `python -m pytest`: 908 passed, the same 2 pre-existing failures (20260801-123345).
- `nix build .#scufris`: passes.
- `python scripts/check_file_size.py`: passes.

## Round 3

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

Both round-2 findings verified CONFIRMED by the same out-of-context reviewer,
which re-ran the R2.1 sabotage on a fresh clone of 6c141b3 (reverting the
condition to `is not None` fails exactly that one test and nothing else) and
re-derived the R2.2 error paths directly: an unwritable state dir still raises
`PermissionError [Errno 13]` and `scufris.db` as a directory still raises
`IsADirectoryError [Errno 21]`, neither relabelled, while a dangling symlink gets
the symlink message. It re-checked all thirteen round-1 findings against the
changed functions and found them still fixed, and confirmed the ContextVar token
is reset on every path it could construct - exception in the body, refused
nesting, and a context manager abandoned without `__exit__`.

One regression from the round-2 fix:

- [x] R3.1 (MINOR) scufris/db/engine.py:136 - the R2.1 guard remembered only the
  INNERMOST open database, so an interleave reopened R1.1's exact failure:
  `with a.transaction(): with b.transaction(): with a.transaction():` was not
  refused, because at the innermost entry the ContextVar held `b`'s path.
  Re-derived in-session: it waits the full busy timeout and dies with
  `OperationalError: database is locked` after 5.01s - the slow, misleading
  failure R1.1 exists to eliminate. Reachable only once a second database exists,
  which is exactly what R2.1 enabled. Make the ContextVar a
  `frozenset[Path]`, refuse on membership, and set the union.
  - Response: fixed. `_open_transactions` is now `ContextVar[frozenset[Path]]`
    defaulting to `frozenset()`; entry refuses on `self._path in open_now` and
    sets `open_now | {self._path}`, so leaving an inner block restores exactly
    the outer set. `test_the_nesting_guard_is_per_database_not_per_process` now
    covers the A > B > A case and asserts it is refused in under a second, so a
    guard that merely renamed the same 5s deadlock would fail it. Falsified:
    setting `frozenset({self._path})` instead of the union fails that test with
    the real `OperationalError`.

### Checks after the round-3 fix

- `ruff check .` and `mypy .`: clean, 172 source files.
- `python -m pytest`: 908 passed, the same 2 pre-existing failures (20260801-123345).
- `nix build .#scufris`: passes.
- `python scripts/check_file_size.py`: passes.

## Round 4

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

R3.1 verified CONFIRMED: A > B > C > A, A > B > A and A > B > B are all refused in
0.00s, the inner-raise path leaves exactly the outer set, and replacing the union
with `frozenset({self._path})` on a clone fails exactly the one test. The
reviewer also confirmed the path-aliasing fix found in-session while it probed
(e668b3b): relative, absolute, `..` and symlinked-state-dir spellings now produce
one key and are refused in 0.00s, and dropping `.resolve()` fails
`test_two_spellings_of_one_database_are_one_database`.

The guard was attacked deliberately this round, because each of its previous two
versions had a hole the next one exposed. It found one more:

- [x] R4.1 (MINOR) scufris/db/engine.py:151 - `_open_transactions.reset(token)`
  restores the set captured at ENTRY, so releasing two nested transactions out of
  LIFO order permanently poisons the context: the second release reinstates a
  path the first had already cleared, and every later unit of work in that thread
  or task is refused with a message naming a transaction that is not open.
  Re-derived in-session with hand-managed `__enter__`/`__exit__`: the guard is
  left holding `{a.path}` and the next `a.transaction()` raises. Not a regression
  from R3.1 - the bool and single-path guards leaked identically - but the set
  makes an order-independent release available. Drop the token and remove in the
  `finally` instead.
  - Response: fixed. The `finally` now does
    `_open_transactions.set(_open_transactions.get() - {self._path})`, which
    restores correctly in ANY release order.
    `test_releasing_out_of_order_does_not_poison_the_context` unwinds outer-first
    and then asserts the next unit of work still commits. Falsified: restoring
    the token-and-reset version fails exactly that test.
- [x] R4.2 (NIT) scufris/db/engine.py:176 - `state_dir.resolve()` dedupes symlinks
  and relative spellings but not two directory trees reaching one inode another
  way; two hardlinked spellings keep distinct keys and slip past the guard.
  Exotic enough that fixing it is optional, but the code should not imply
  resolution is complete.
  - Response: fixed by documenting the boundary rather than keying on
    `(st_dev, st_ino)`. The resolve comment now says the key is the resolved PATH
    and that two hardlinked spellings are out of scope for a single-host app that
    opens exactly one database. Keying on the inode would buy a case nothing in
    this system can reach.

### Checks after the round-4 fixes

- `ruff check .` and `mypy .`: clean, 172 source files.
- `python -m pytest`: 910 passed, the same 2 pre-existing failures (20260801-123345).
- `nix build .#scufris`: passes.
- `python scripts/check_file_size.py`: passes.

### On the four rounds this guard took

The nesting guard was rewritten four times and each version's hole was found by
attacking the previous one, not by reading it: process-global refused two
databases that could not contend; per-database missed A inside B inside A; the
resolved path missed two spellings of one file; the entry-snapshot reset poisoned
a non-LIFO unwind. Every version passed its own tests. What actually converged it
was writing the sabotage first and asking which mutation the suite would let
through - and the last two holes were found by the reviewer being told to attack
the guard specifically rather than to verify it.

## Round 5

- REVIEWER: out-of-context
- VERDICT: APPROVE

Both round-4 findings verified CONFIRMED. R4.1's sabotage was re-run on a clone
at HEAD: restoring the token-and-reset version fails exactly
`test_releasing_out_of_order_does_not_poison_the_context` and nothing else. R4.2
was judged on its merits rather than waved through - documenting the boundary is
the proportionate resolution, because reaching it needs someone to hardlink
`scufris.db` into a second state dir and open both, which the app neither does
nor can be led into, and keying on `(st_dev, st_ino)` would add a stat and a
second notion of identity to buy nothing.

No new findings. The reviewer attacked the guard rather than reading it, and it
held under: all six release orders of three nested databases; 200 randomized
unwinds throwing into levels in random order; `ExitStack`; a double `__exit__` on
an exhausted context manager; 20 trials of four abandoned context managers
collected by the GC in shuffled order; `close()` mid-transaction; a `begin()` that
fails outright; and eight threads plus five `asyncio.to_thread` offloads on one
database concurrently. In every case the set returned to `frozenset()` and no
false refusal followed. The only refusals were the intended ones - the same
resolved path in one context, including an offload from inside a transaction on
that database.

### Checks at the approved commit

- `ruff check .` and `mypy .`: clean, 172 source files.
- `python -m pytest`: 912 collected, 910 passed, 2 failed - the pre-existing
  `test_project_tasks_endpoint` and `test_read_project_tasks_parses_real_tatr`,
  both reproducing identically on master and filed as 20260801-123345.
- `nix build .#scufris`: passes (run by the primary; the lanes were excluded from
  it throughout).
- `python scripts/check_file_size.py`: passes. `engine.py` 250 lines,
  `test_db_engine.py` 465, no ALLOWLIST change.

### Pending user checks

None. This task has no `manual:` proof.
