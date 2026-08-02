"""The transaction boundary, proven on its own before any store sits on it.

Every proof here runs against a SCRATCH table this module creates. The core owns
no product schema - `models.py` and `migrations/` arrive in the follow-up tasks -
so proving the boundary means proving it against a table the boundary itself can
create inside a transaction.

The interesting proofs are the ones a hand-rolled connection passes and a POOL
fails: a pragma applied once at open (rather than per connection), and pysqlite's
implicit deferred begin (which is silently accepted and only fails under
contention). Both get a test that discriminates, not one that restates the code.
"""

from __future__ import annotations

import asyncio
import sqlite3
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError

from scufris.db import DATABASE_FILENAME, Database, open_database

CREATE_SCRATCH = (
    "CREATE TABLE IF NOT EXISTS scratch (id INTEGER PRIMARY KEY, n INTEGER)"
)


def _scratch(db: Database) -> None:
    with db.transaction() as conn:
        conn.execute(text(CREATE_SCRATCH))


def _rows(db: Database) -> list[int]:
    with db.transaction() as conn:
        return [
            row[0] for row in conn.execute(text("SELECT n FROM scratch ORDER BY n"))
        ]


# --------------------------------------------------------------------------
# The unit of work
# --------------------------------------------------------------------------


def test_state_transaction_rolls_back_as_a_unit(database: Database) -> None:
    """A failed multi-statement unit of work commits NOTHING - not even the
    statements that succeeded before the failure."""
    _scratch(database)

    with pytest.raises(RuntimeError, match="halfway"):
        with database.transaction() as conn:
            conn.execute(text("INSERT INTO scratch (n) VALUES (1)"))
            conn.execute(text("INSERT INTO scratch (n) VALUES (2)"))
            raise RuntimeError("halfway through the unit of work")

    assert _rows(database) == []


def test_sync_and_async_callers_share_the_transaction_boundary(
    database: Database,
) -> None:
    """Thread-pool callers and loop callbacks offloading through
    ``asyncio.to_thread`` contend on ONE boundary without losing a write.

    Each worker does a read-modify-write inside the transaction, which is the
    shape that loses data when the lock is around the persist rather than around
    the read. The count is the proof: 2 x 20 increments, no lost update.
    """
    _scratch(database)
    with database.transaction() as conn:
        conn.execute(text("INSERT INTO scratch (id, n) VALUES (1, 0)"))

    rounds = 20

    def increment() -> None:
        with database.transaction() as conn:
            current = conn.execute(
                text("SELECT n FROM scratch WHERE id = 1")
            ).scalar_one()
            conn.execute(
                text("UPDATE scratch SET n = :n WHERE id = 1"), {"n": current + 1}
            )

    async def from_the_loop() -> None:
        await asyncio.gather(*(asyncio.to_thread(increment) for _ in range(rounds)))

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(increment) for _ in range(rounds)]
        asyncio.run(from_the_loop())
        for future in futures:
            future.result()

    with database.transaction() as conn:
        total = conn.execute(text("SELECT n FROM scratch WHERE id = 1")).scalar_one()
    assert total == rounds * 2


def test_nested_transactions_are_refused_immediately(database: Database) -> None:
    """Re-entry raises at once instead of deadlocking the caller against itself.

    Without the guard the inner ``BEGIN IMMEDIATE`` checks out a SECOND pooled
    connection and waits the full 5s ``busy_timeout`` on the write lock the outer
    transaction is holding, then fails with a message that reads as external
    contention. The elapsed time is part of the assertion: a guard that merely
    turned the same deadlock into a different exception would still cost 5s.
    """
    _scratch(database)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="do not nest"):
        with database.transaction():
            with database.transaction():
                pass
    assert time.monotonic() - started < 1.0

    # The guard releases with the outer block, so the next unit of work is fine.
    with database.transaction() as conn:
        conn.execute(text("INSERT INTO scratch (n) VALUES (1)"))
    assert _rows(database) == [1]


def test_the_nesting_guard_is_per_database_not_per_process(tmp_path: Path) -> None:
    """Two units of work on two different FILES are not the deadlock being named.

    They cannot contend on each other's write lock, so refusing them would be a
    false positive with a message that misstates the cause. A guard keyed on a
    bare bool rather than on the database's path fails this.
    """
    first = open_database(tmp_path / "one")
    second = open_database(tmp_path / "two")
    try:
        with first.transaction() as conn:
            conn.execute(text(CREATE_SCRATCH))
            with second.transaction() as other:
                other.execute(text(CREATE_SCRATCH))
                other.execute(text("INSERT INTO scratch (n) VALUES (2)"))
            conn.execute(text("INSERT INTO scratch (n) VALUES (1)"))

        assert _rows(first) == [1]
        assert _rows(second) == [2]

        # ... but interleaving back onto the FIRST database is the real deadlock
        # again, so it is refused, and refused immediately. A guard that
        # remembered only the innermost database would let this through and wait
        # the full busy timeout on a lock it is holding itself.
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="do not nest"):
            with first.transaction():
                with second.transaction():
                    with first.transaction():
                        pass
        assert time.monotonic() - started < 1.0
    finally:
        first.close()
        second.close()


def test_releasing_out_of_order_does_not_poison_the_context(tmp_path: Path) -> None:
    """Unwinding two transactions out of LIFO order leaves the guard clean.

    Hand-managed context managers and GC-finalized generators both release out of
    order. A guard that restored its ENTRY snapshot would reinstate a path the
    other release had already cleared, and then refuse every later unit of work
    in this thread with a message naming a transaction that is not open.
    """
    first = open_database(tmp_path / "one")
    second = open_database(tmp_path / "two")
    try:
        outer = first.transaction()
        inner = second.transaction()
        outer.__enter__()
        inner.__enter__()
        outer.__exit__(None, None, None)  # the OUTER one first
        inner.__exit__(None, None, None)

        # Nothing is open now, so this must simply work.
        with first.transaction() as conn:
            conn.execute(text(CREATE_SCRATCH))
            conn.execute(text("INSERT INTO scratch (n) VALUES (1)"))
        assert _rows(first) == [1]
    finally:
        first.close()
        second.close()


def test_two_spellings_of_one_database_are_one_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard's key is the FILE, not the string the caller happened to type.

    A relative and an absolute spelling of the same state dir are the same
    database and the same write lock, so nesting them is the deadlock. Without
    resolving the directory the two keys differ, the guard misses, and the caller
    waits the full busy timeout on a lock it is holding itself.
    """
    (tmp_path / "state").mkdir()
    monkeypatch.chdir(tmp_path)

    relative = open_database(Path("state"))
    absolute = open_database(tmp_path / "state")
    try:
        assert relative.path == absolute.path

        started = time.monotonic()
        with pytest.raises(RuntimeError, match="do not nest"):
            with relative.transaction():
                with absolute.transaction():
                    pass
        assert time.monotonic() - started < 1.0
    finally:
        relative.close()
        absolute.close()


def test_the_nesting_guard_is_per_context_not_global(database: Database) -> None:
    """One caller's open transaction must not refuse another caller's.

    The guard is a ``ContextVar``, so a worker thread starts with its own copy.
    A guard hoisted to module state would make this test fail while the
    single-threaded nesting test above still passed.
    """
    _scratch(database)
    with ThreadPoolExecutor(max_workers=2) as pool:

        def unit(n: int) -> None:
            with database.transaction() as conn:
                conn.execute(text("INSERT INTO scratch (n) VALUES (:n)"), {"n": n})

        list(pool.map(unit, (1, 2)))

    assert _rows(database) == [1, 2]


# --------------------------------------------------------------------------
# What a pool breaks
# --------------------------------------------------------------------------


def test_every_pooled_connection_applies_production_pragmas(
    database: Database,
) -> None:
    """EVERY pooled connection carries the four pragmas, not just the first.

    Two connections are held open at once so the pool is forced to dial a second
    one; applying the pragmas at open rather than on the ``connect`` event passes
    this test on connection one and fails it on connection two.
    """
    expected: dict[str, object] = {
        "journal_mode": "wal",
        "synchronous": 2,  # FULL
        "busy_timeout": 5000,
        "foreign_keys": 1,
    }

    def read_pragmas(raw: object) -> dict[str, object]:
        # Read through the DBAPI connection, not through ``Connection.execute``:
        # the latter opens a transaction, and every begin on this engine is a
        # BEGIN IMMEDIATE, so the two would serialise on the write lock and the
        # test would prove contention instead of pragmas.
        cursor = raw.cursor()  # type: ignore[attr-defined]
        try:
            return {
                name: cursor.execute(f"PRAGMA {name}").fetchone()[0]
                for name in expected
            }
        finally:
            cursor.close()

    first = database.engine.raw_connection()
    second = database.engine.raw_connection()
    try:
        assert first.dbapi_connection is not second.dbapi_connection
        observed = [read_pragmas(first), read_pragmas(second)]
    finally:
        first.close()
        second.close()

    for pragmas in observed:
        assert str(pragmas["journal_mode"]).lower() == expected["journal_mode"]
        assert pragmas["synchronous"] == expected["synchronous"]
        assert pragmas["busy_timeout"] == expected["busy_timeout"]
        assert pragmas["foreign_keys"] == expected["foreign_keys"]


def test_transaction_uses_begin_immediate(database: Database) -> None:
    """The transaction takes the write lock UP FRONT.

    Discriminating probe: a second connection with a zero busy timeout tries to
    take the write lock while an open transaction has issued no statement at all.
    Under ``BEGIN IMMEDIATE`` the lock is already held and the probe is refused;
    under pysqlite's implicit deferred begin nothing is locked yet and the probe
    succeeds. A test that only asserted the hook fired would pass either way.
    """
    _scratch(database)

    probe = sqlite3.connect(database.path, timeout=0, isolation_level=None)
    try:
        with database.transaction():
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                probe.execute("BEGIN IMMEDIATE")
        # ... and the lock is released with the transaction.
        probe.execute("BEGIN IMMEDIATE")
        probe.execute("ROLLBACK")
    finally:
        probe.close()


# --------------------------------------------------------------------------
# Opening the file
# --------------------------------------------------------------------------


def test_damaged_state_refuses_to_load(tmp_path: Path) -> None:
    """A damaged database raises instead of presenting itself as EMPTY.

    An empty-looking store is the failure that silently discards an operator's
    records; there is no tolerant loader anywhere in this package.
    """
    (tmp_path / DATABASE_FILENAME).write_bytes(b"this is not a SQLite database\n" * 64)

    with pytest.raises(DatabaseError):
        open_database(tmp_path)


def test_corrupt_pages_behind_a_good_header_raise_at_the_first_read(
    tmp_path: Path,
) -> None:
    """The other half of damaged-is-not-empty, and the honest limit of it.

    SQLite validates a page when it reaches it, so a file whose page-1 header is
    intact and whose later pages are zeroed OPENS cleanly - only the read raises.
    What matters is what the caller can never see: an empty store. It gets a
    ``DatabaseError`` either way, and the docs promise exactly that rather than
    claiming open catches everything.
    """
    first = open_database(tmp_path)
    try:
        _scratch(first)
        with first.transaction() as conn:
            for n in range(500):
                conn.execute(text("INSERT INTO scratch (n) VALUES (:n)"), {"n": n})
    finally:
        first.close()

    path = tmp_path / DATABASE_FILENAME
    raw = bytearray(path.read_bytes())
    assert len(raw) > 4096, "the fixture needs pages past the header to corrupt"
    raw[4096:] = b"\x00" * (len(raw) - 4096)
    path.write_bytes(bytes(raw))

    database = open_database(tmp_path)
    try:
        with pytest.raises(DatabaseError):
            _rows(database)
    finally:
        database.close()


def test_a_symlinked_database_path_is_refused(tmp_path: Path) -> None:
    """The state dir is a directory, not a promise.

    Without ``O_NOFOLLOW`` the database is initialized THROUGH the link and the
    0600 chmod lands on a file outside the state directory.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "victim"
    target.write_bytes(b"")
    target.chmod(0o644)

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / DATABASE_FILENAME).symlink_to(target)

    with pytest.raises(RuntimeError, match="symlink"):
        open_database(state_dir)
    assert stat.S_IMODE(target.stat().st_mode) == 0o644, "the target was touched"
    assert target.read_bytes() == b"", "the target was written through"


def test_sidecars_left_behind_by_a_crash_are_narrowed_on_open(tmp_path: Path) -> None:
    """The sibling half of the chmod earns its place on the CRASH path.

    On a fresh open SQLite has not created ``-wal``/``-shm`` yet and inherits the
    database's mode when it does. The case that needs the loop is the one where a
    previous run died leaving sidecars behind under a laxer umask: nothing else in
    the system would ever narrow them.
    """
    base = tmp_path / DATABASE_FILENAME
    leftovers = [Path(f"{base}-wal"), Path(f"{base}-shm")]
    for path in leftovers:
        path.write_bytes(b"")
        path.chmod(0o644)

    database = open_database(tmp_path)
    try:
        for path in leftovers:
            assert stat.S_IMODE(path.stat().st_mode) == 0o600, path.name
    finally:
        database.close()


def test_state_database_files_are_owner_only(tmp_path: Path) -> None:
    """The database and its ``-wal``/``-shm`` siblings are owner-only.

    The boundary will hold auth session identifiers, which ``auth/store.py``
    protects the same way. The siblings matter as much as the database: a
    committed row lives in the ``-wal`` file until the next checkpoint.
    """
    database = open_database(tmp_path)
    try:
        _scratch(database)
        with database.transaction() as conn:
            conn.execute(text("INSERT INTO scratch (n) VALUES (7)"))

        base = tmp_path / DATABASE_FILENAME
        siblings = [base, Path(f"{base}-wal"), Path(f"{base}-shm")]
        for path in siblings:
            assert path.exists(), f"{path.name} was never created"
            assert stat.S_IMODE(path.stat().st_mode) == 0o600, path.name
    finally:
        database.close()


def test_reopening_the_file_sees_committed_rows(tmp_path: Path) -> None:
    """Restart survival: the rows are in the FILE, not in a process mirror."""
    first = open_database(tmp_path)
    try:
        _scratch(first)
        with first.transaction() as conn:
            conn.execute(text("INSERT INTO scratch (n) VALUES (42)"))
    finally:
        first.close()

    second = open_database(tmp_path)
    try:
        assert _rows(second) == [42]
    finally:
        second.close()


def test_transaction_is_usable_from_a_worker_thread(database: Database) -> None:
    """One engine, many threads: the pool hands each thread its own connection."""
    _scratch(database)
    seen: list[int] = []
    lock = threading.Lock()

    def work(n: int) -> None:
        with database.transaction() as conn:
            conn.execute(text("INSERT INTO scratch (n) VALUES (:n)"), {"n": n})
        with lock:
            seen.append(n)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(10)))

    assert sorted(seen) == list(range(10))
    assert _rows(database) == list(range(10))


def test_open_waits_out_a_concurrent_first_wal_conversion(tmp_path: Path) -> None:
    """Two processes opening a FRESH state dir at once: neither dies at open.

    The one pragma `busy_timeout` does not cover. SQLite refuses a journal-mode
    change while another connection holds the write lock and returns SQLITE_BUSY
    WITHOUT running the busy handler, so before the retry in `_set_journal_mode`
    this raised "database is locked" immediately - measured, in 0.000s against a
    5s timeout. Only reachable on the one-time delete->WAL conversion, which is
    exactly what a first startup does.

    The holder is a plain `sqlite3` connection, not a second `Database`: opening
    one through `open_database` would itself convert the file to WAL and there
    would be nothing left to race on.
    """
    path = tmp_path / DATABASE_FILENAME
    # check_same_thread=False so the releaser thread below can commit on it;
    # nothing else touches it concurrently.
    holder = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    holder.execute("PRAGMA busy_timeout=5000")
    assert holder.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("CREATE TABLE held (x)")

    released = threading.Event()

    def release() -> None:
        time.sleep(0.3)
        holder.execute("COMMIT")
        released.set()

    releaser = threading.Thread(target=release)
    releaser.start()
    try:
        db = open_database(tmp_path)
        try:
            assert released.is_set(), "open returned before the lock was released"
            with db.transaction() as conn:
                mode = conn.exec_driver_sql("PRAGMA journal_mode").scalar()
            assert mode == "wal"
        finally:
            db.close()
    finally:
        releaser.join()
        holder.close()


def test_transaction_refuses_the_event_loop_thread(database: Database) -> None:
    """A transaction opened from a thread with a running loop raises at once.

    The boundary holds SQLite's single write lock, so opening one ON the loop
    stalls every other writer for as long as the unit of work runs - measured at
    3.04s against a 0.01s heartbeat in 20260801-120412. That is a rule prose
    cannot enforce: the failure is a latency regression on an unrelated request,
    which no store's own test would ever see. The message names the offload the
    caller is supposed to use, because "you are on the loop" is not actionable on
    its own.

    ``asyncio.to_thread`` from the SAME coroutine still works: the worker thread
    has no loop of its own, so the offload the message prescribes is the thing
    the guard lets through.
    """
    _scratch(database)

    async def from_the_loop() -> None:
        with pytest.raises(RuntimeError, match="asyncio.to_thread"):
            with database.transaction():
                pass

        def unit_of_work() -> None:
            with database.transaction() as conn:
                conn.execute(text("INSERT INTO scratch (n) VALUES (7)"))

        await asyncio.to_thread(unit_of_work)

    asyncio.run(from_the_loop())
    assert _rows(database) == [7]
