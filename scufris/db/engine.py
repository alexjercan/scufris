"""The one transactional boundary every app-owned store will sit on.

One SQLite database at ``<state_dir>/scufris.db``, reached through one
synchronous context manager, ``Database.transaction()``. There is no async
engine, no second store API and no in-memory mirror: a committed row is in the
file, and the file is the truth.

Three things here exist because a CONNECTION POOL is not a hand-rolled
connection, and each has its own test:

- The four pragmas are applied on the ``connect`` EVENT, so every connection the
  pool ever dials carries them. Applying them once at open configures connection
  one and silently leaves connection two on SQLite's defaults - rollback journal,
  ``synchronous=NORMAL``, no busy timeout, foreign keys off.
- The begin is ``BEGIN IMMEDIATE``, not pysqlite's implicit deferred begin. A
  deferred begin takes only a read lock, so two concurrent read-modify-write
  transactions both read, then both try to upgrade, and one is refused with
  SQLITE_BUSY that ``busy_timeout`` does NOT retry. Taking the write lock up
  front turns that into a wait instead of an error.
- The driver's own implicit transaction handling is turned off
  (``isolation_level=None``) so SQLAlchemy's begin is the only begin.

Rules for callers:

- A transaction NEVER spans an ``await``. It holds SQLite's single write lock,
  so suspending inside one blocks every other writer on whatever the awaited
  thing is waiting for.
- Loop-thread callers therefore wrap a SYNCHRONOUS unit of work and offload it:
  ``await asyncio.to_thread(unit_of_work)``, where ``unit_of_work`` opens and
  closes the transaction inside the worker thread.
- The transaction is the read-modify-write boundary. Read inside it, not before
  it; a lock around only the persist loses the update it read outside.
- Because every begin is immediate, a read-only unit of work also takes the
  write lock. That is the accepted cost of having ONE boundary rather than two;
  keep units of work short rather than adding a read-only variant.
- A unit of work NEVER nests. Pass the open ``Connection`` down to whatever the
  step needs; re-entering ``transaction()`` raises. Without that guard the
  natural mistake - one store's unit of work calling another store's - checks
  out a second pooled connection, waits the full busy timeout on the write lock
  the OUTER transaction is holding, and then fails in a way that reads as
  external contention rather than as the bug it is.

The schema itself is NOT here. ``models.py`` and ``migrations/`` arrive in the
following tasks; this module knows how to open a database and how to write to it
safely, and nothing about what is in it.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from sqlalchemy import Connection, Engine, create_engine, event

DATABASE_FILENAME = "scufris.db"

# SQLite writes committed data to these two siblings before the next checkpoint
# folds it back, so they need the database's own permissions, not the umask's.
SIDECAR_SUFFIXES = ("-wal", "-shm")

# 0600, matching auth/store.py: this boundary will hold live session ids.
FILE_MODE = 0o600

# WAL so a reader never blocks the writer; FULL because the machine this runs on
# loses power; a busy timeout so contention waits instead of raising; foreign
# keys because SQLite leaves them OFF per connection.
#
# busy_timeout goes FIRST: the journal_mode change can itself contend, and until
# this line runs the only timeout in force is the driver's own connect default.
PRAGMAS: tuple[str, ...] = (
    "PRAGMA busy_timeout=5000",
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=FULL",
    "PRAGMA foreign_keys=ON",
)

# Which databases THIS context already holds a transaction on. A ContextVar
# rather than a threading.local because ``asyncio.to_thread`` copies the calling
# context into the worker thread, so the guard follows the offload the way the
# callers are told to write it.
#
# A SET of paths, not one path: transactions on two different FILES cannot
# contend and so are not the deadlock this guards, but remembering only the
# innermost would let A inside B inside A through - and that one really does
# deadlock, on A's own write lock.
_open_transactions: ContextVar[frozenset[Path]] = ContextVar(
    "scufris_db_open_transactions", default=frozenset()
)


def database_path(state_dir: Path) -> Path:
    """Where the one database lives under a state directory."""
    return state_dir / DATABASE_FILENAME


class Database:
    """An open SQLite database and the single way to write to it.

    Construct through :func:`open_database`.
    """

    def __init__(self, engine: Engine, path: Path) -> None:
        self._engine = engine
        self._path = path

    @property
    def engine(self) -> Engine:
        """The configured engine, for Alembic and for declarative metadata.

        Callers that WRITE use :meth:`transaction` instead - it is the only place
        that owns a begin and a commit.
        """
        return self._engine

    @property
    def path(self) -> Path:
        """The database file itself."""
        return self._path

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        """One unit of work: begins immediately, commits on exit, rolls back whole.

        Everything inside is one atomic step. A statement that succeeded before a
        later failure is rolled back with the rest - there is no partial commit
        and no savepoint API on this surface.

        Synchronous by design. From the event loop, call it inside a function
        handed to ``asyncio.to_thread``; never hold it across an ``await``.

        Raises ``RuntimeError`` on re-entry rather than deadlocking against
        itself. Nesting is never made to work silently: an inner block that
        appears to commit but does not is worse than the error it replaces.
        """
        open_now = _open_transactions.get()
        if self._path in open_now:
            raise RuntimeError(
                f"a transaction on {self._path} is already open in this context; "
                "units of work do not nest - pass the open Connection down "
                "instead of calling transaction() again"
            )
        _open_transactions.set(open_now | {self._path})
        try:
            with self._engine.begin() as conn:
                yield conn
        finally:
            # Remove this path rather than resetting to the entry snapshot. A
            # reset restores the set as it was at ENTRY, so releasing two nested
            # transactions out of LIFO order - which hand-managed context
            # managers and GC-finalized generators both do - would reinstate a
            # path the other release had already cleared and refuse every later
            # unit of work in this context. A difference is order-independent.
            _open_transactions.set(_open_transactions.get() - {self._path})

    def close(self) -> None:
        """Return every pooled connection. The file stays where it is."""
        self._engine.dispose()


def open_database(state_dir: Path) -> Database:
    """Open (creating if absent) the one state database under ``state_dir``.

    DAMAGE IS NEVER PRESENTED AS EMPTY. It surfaces as
    ``sqlalchemy.exc.DatabaseError``, at open when the header itself is
    unreadable and at the first read when the header is intact but the pages
    behind it are not - SQLite validates a page when it reaches it, and a
    ``quick_check`` of the whole file at every startup is not worth its cost.
    Either way the caller gets an exception, never an empty store. There is no
    tolerant loader in this package.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    # Resolve the DIRECTORY, not the database path: the path is the key the
    # nesting guard compares, so two spellings of the same file - relative and
    # absolute, or through a symlinked parent - have to produce one key or the
    # guard misses the deadlock it exists to name. Resolving the database path
    # itself would also resolve a symlinked final component and defeat the
    # O_NOFOLLOW below, which is the one thing that must NOT be followed.
    # The key is the resolved PATH: two hardlinked spellings of one inode would
    # still read as two databases, which is out of scope for a single-host app
    # that opens exactly one.
    path = database_path(state_dir.resolve())
    # Create the file ourselves at 0600 rather than letting SQLite create it
    # under the umask: SQLite copies the database's mode onto the -wal/-shm it
    # creates later, so getting this right once covers all three. O_NOFOLLOW
    # because the state dir is a directory, not a promise: opening the database
    # THROUGH a symlink would initialize, and then chmod, a file elsewhere.
    try:
        os.close(os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, FILE_MODE))
    except OSError as exc:
        if path.is_symlink():
            raise RuntimeError(
                f"{path} is a symlink; refusing to open the database through it"
            ) from exc
        raise

    engine = create_engine(
        f"sqlite:///{path}",
        # Hand pysqlite's implicit transaction handling to SQLAlchemy, so the
        # "begin" event below is the only thing that opens a transaction.
        connect_args={"isolation_level": None},
    )

    @event.listens_for(engine, "connect")
    def _apply_pragmas(dbapi_connection: sqlite3.Connection, _record: object) -> None:
        cursor = dbapi_connection.cursor()
        try:
            for pragma in PRAGMAS:
                cursor.execute(pragma)
        finally:
            cursor.close()

    @event.listens_for(engine, "begin")
    def _begin_immediate(conn: Connection) -> None:
        conn.exec_driver_sql("BEGIN IMMEDIATE")

    # Dial one connection now: this runs the pragma hook, which is what turns an
    # unreadable header into a DatabaseError at open rather than at whichever
    # read happens to come first, and it is what puts the database into WAL mode.
    try:
        with engine.connect():
            pass
        _secure(path)
    except Exception:
        engine.dispose()
        raise

    return Database(engine, path)


def _secure(path: Path) -> None:
    """Force 0600 on the database and on whichever siblings already exist.

    On a fresh open only the database is there: SQLite creates ``-wal`` and
    ``-shm`` at the first write, with the database's own mode. The sibling half
    of this loop is for the other case - sidecars a crashed run left behind under
    a laxer umask, which nothing else would ever narrow.
    """
    for candidate in (path, *(Path(f"{path}{suffix}") for suffix in SIDECAR_SUFFIXES)):
        if candidate.is_symlink():
            raise RuntimeError(
                f"{candidate} is a symlink; refusing to chmod its target"
            )
        try:
            os.chmod(candidate, FILE_MODE)
        except FileNotFoundError:
            continue
