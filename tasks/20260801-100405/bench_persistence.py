#!/usr/bin/env python3
"""Measure SQLite against locked atomic JSON snapshots on the axes the
successor decision turns on.

The predecessor spike (20260729-102146) proved the CURRENT stores race. This
script does not re-prove that. It compares the two candidate REPLACEMENTS, both
implemented as well as each can be:

  * ``JsonStore``   - one file per store, a per-store ``threading.RLock`` held
                      across read-modify-write, a UNIQUE temp name, ``fsync``
                      of both the temp file and the containing directory.
                      This is the strongest form of the incumbent discipline;
                      the predecessor's control showed the weak form is unsafe.
  * ``SqliteStore`` - stdlib ``sqlite3``, WAL, ``synchronous=FULL``,
                      ``busy_timeout``, one connection per thread, explicit
                      ``BEGIN IMMEDIATE`` transactions. No ORM, no new
                      dependency.

Scenarios, each printing numbers rather than opinions:

  1 race        concurrent single-store writes from an OS thread pool
  2 multi       a three-record terminal state committed with a failure injected
                between records
  3 events      append-only activity events: cost per append, pagination,
                retention delete
  4 crash       SIGKILL a writer mid-flight, then reopen the store
  5 procs       two PROCESSES writing the same store at once
  6 migrate     legacy-JSON import: idempotency, backup, corrupt-input
                diagnostics
  7 isolation   per-test setup cost of an empty store
  8 readers     read latency while writers hammer a warm store
  9 leftovers   what a SIGKILL leaves in the state directory
  10 asyncio    event-loop lag when a commit runs on the loop thread

Run everything:

    nix develop --command python tasks/20260801-100405/bench_persistence.py

Run one scenario:

    nix develop --command python tasks/20260801-100405/bench_persistence.py events

Exit code is 0 whenever the run COMPLETES; this is a measurement harness, not a
test. Read the numbers.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

WRITERS = 8
ROUNDS = 25
PAD = "x" * 4096  # one record spans several write(2) calls, as a real store does


# --------------------------------------------------------------------------
# Candidate A: locked atomic JSON snapshot
# --------------------------------------------------------------------------


class JsonStore:
    """A snapshot store written as safely as the JSON discipline allows."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._rows: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        # Deliberately NOT tolerant: a damaged store raises instead of
        # presenting itself as empty. See constraint 6 of the predecessor.
        self._rows = json.loads(self._path.read_text())

    def _persist(self) -> None:
        # Unique temp name, fsync of the file AND the directory, then replace.
        tmp = self._path.with_name(
            f"{self._path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        with open(tmp, "w") as handle:
            json.dump(self._rows, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self._path)
        dir_fd = os.open(self._path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def create(self, key: str, row: dict[str, Any]) -> None:
        with self._lock:
            before = dict(self._rows)
            self._rows[key] = row
            try:
                self._persist()
            except Exception:
                self._rows = before  # commit-or-revert, constraint 4
                raise

    def count(self) -> int:
        with self._lock:
            return len(self._rows)


# --------------------------------------------------------------------------
# Candidate B: SQLite
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS rows (
    key   TEXT PRIMARY KEY,
    body  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY, state TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    agent_id TEXT PRIMARY KEY, session_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    agent_id TEXT PRIMARY KEY, verdict TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id TEXT NOT NULL,
    kind           TEXT NOT NULL,
    body           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_correlation ON events(correlation_id, seq);
CREATE TABLE IF NOT EXISTS deliveries (
    channel        TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    PRIMARY KEY (channel, idempotency_key)
);
"""


class SqliteStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._local = threading.local()
        conn = self.conn
        conn.executescript(SCHEMA)
        conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def create(self, key: str, row: dict[str, Any]) -> None:
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO rows(key, body) VALUES (?, ?)", (key, json.dumps(row))
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def count(self) -> int:
        return int(self.conn.execute("SELECT count(*) FROM rows").fetchone()[0])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _tmpdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="scufris-bench-"))


def _hammer(
    fn: Callable[[int, int], None], writers: int, rounds: int
) -> list[BaseException]:
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(w: int) -> None:
        for r in range(rounds):
            try:
                fn(w, r)
            except BaseException as exc:  # noqa: BLE001 - counting them is the point
                with lock:
                    errors.append(exc)

    with ThreadPoolExecutor(max_workers=writers) as pool:
        list(pool.map(worker, range(writers)))
    return errors


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


# --------------------------------------------------------------------------
# 1. concurrent single-store writes
# --------------------------------------------------------------------------


def scenario_race() -> None:
    _section(f"1 race: {WRITERS} threads x {ROUNDS} creates into ONE store")
    expected = WRITERS * ROUNDS

    for name, factory, reopen in (
        (
            "locked JSON",
            lambda d: JsonStore(d / "s.json"),
            lambda d: JsonStore(d / "s.json"),
        ),
        (
            "sqlite",
            lambda d: SqliteStore(d / "s.db"),
            lambda d: SqliteStore(d / "s.db"),
        ),
    ):
        d = _tmpdir()
        store = factory(d)
        started = time.perf_counter()

        def one_write(w: int, r: int, store: Any = store) -> None:
            store.create(f"{w}-{r}", {"w": w, "r": r, "pad": PAD})

        errors = _hammer(one_write, WRITERS, ROUNDS)
        elapsed = time.perf_counter() - started
        after = reopen(d).count()
        print(
            f"  {name:<12} expected={expected} after_restart={after} "
            f"raised={len(errors)} wall={elapsed:.3f}s "
            f"per_write={elapsed / expected * 1000:.2f}ms"
        )
        if errors:
            print(f"    first error: {type(errors[0]).__name__}: {errors[0]}")
        shutil.rmtree(d)


# --------------------------------------------------------------------------
# 2. three-record terminal state with an injected mid-commit failure
# --------------------------------------------------------------------------


class _Boom(RuntimeError):
    pass


def scenario_multi() -> None:
    _section("2 multi: agent+session+outcome, failure injected before the 3rd record")
    runs = 200
    fail_every = 2  # half the commits are interrupted

    # --- locked JSON: three files, one global lock, no cross-file transaction
    d = _tmpdir()
    agents = JsonStore(d / "agents.json")
    sessions = JsonStore(d / "sessions.json")
    outcomes = JsonStore(d / "outcomes.json")
    glock = threading.RLock()
    json_failed = 0
    for i in range(runs):
        aid = f"a{i}"
        try:
            with glock:
                agents.create(aid, {"state": "finished"})
                sessions.create(aid, {"session_id": f"s{i}"})
                if i % fail_every == 0:
                    raise _Boom("crash between records")
                outcomes.create(aid, {"verdict": "ok"})
        except _Boom:
            json_failed += 1
    a2 = JsonStore(d / "agents.json")
    s2 = JsonStore(d / "sessions.json")
    o2 = JsonStore(d / "outcomes.json")
    torn_json = s2.count() - o2.count()
    print(
        f"  locked JSON  interrupted={json_failed} agents={a2.count()} "
        f"sessions={s2.count()} outcomes={o2.count()} torn(session_no_outcome)={torn_json}"
    )
    shutil.rmtree(d)

    # --- sqlite: one transaction over all three tables
    d = _tmpdir()
    store = SqliteStore(d / "s.db")
    conn = store.conn
    sql_failed = 0
    for i in range(runs):
        aid = f"a{i}"
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO agents(id, state) VALUES (?, ?)", (aid, "finished")
            )
            conn.execute(
                "INSERT INTO sessions(agent_id, session_id) VALUES (?, ?)",
                (aid, f"s{i}"),
            )
            if i % fail_every == 0:
                raise _Boom("crash between records")
            conn.execute(
                "INSERT INTO outcomes(agent_id, verdict) VALUES (?, ?)", (aid, "ok")
            )
            conn.execute("COMMIT")
        except _Boom:
            conn.execute("ROLLBACK")
            sql_failed += 1
    fresh = SqliteStore(d / "s.db").conn
    na = fresh.execute("SELECT count(*) FROM agents").fetchone()[0]
    ns = fresh.execute("SELECT count(*) FROM sessions").fetchone()[0]
    no = fresh.execute("SELECT count(*) FROM outcomes").fetchone()[0]
    torn_sql = fresh.execute(
        "SELECT count(*) FROM sessions LEFT JOIN outcomes USING (agent_id) "
        "WHERE outcomes.agent_id IS NULL"
    ).fetchone()[0]
    print(
        f"  sqlite       interrupted={sql_failed} agents={na} "
        f"sessions={ns} outcomes={no} torn(session_no_outcome)={torn_sql}"
    )
    shutil.rmtree(d)


# --------------------------------------------------------------------------
# 3. append-only activity events
# --------------------------------------------------------------------------


def scenario_events(total: int = 5000) -> None:
    _section(f"3 events: {total} ordered append-only events, then paginate + retain")

    # --- locked JSON: appending re-serializes the whole list every time
    d = _tmpdir()
    path = d / "events.json"
    store = JsonStore(path)
    latencies: list[float] = []
    started = time.perf_counter()
    for i in range(total):
        t0 = time.perf_counter()
        store.create(
            f"{i:08d}",
            {"correlation_id": f"c{i % 50}", "kind": "run", "body": PAD[:200]},
        )
        latencies.append((time.perf_counter() - t0) * 1000)
    json_wall = time.perf_counter() - started
    json_bytes = path.stat().st_size
    json_first = statistics.mean(latencies[:100])
    json_last = statistics.mean(latencies[-100:])
    t0 = time.perf_counter()
    rows = json.loads(path.read_text())
    page = sorted(rows.items())[4000:4050]
    json_page = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    keep = dict(sorted(rows.items())[-1000:])
    store._rows = keep
    store._persist()
    json_retain = (time.perf_counter() - t0) * 1000
    print(
        f"  locked JSON  wall={json_wall:.2f}s bytes={json_bytes:,} "
        f"append_first100={json_first:.2f}ms append_last100={json_last:.2f}ms "
        f"(x{json_last / json_first:.1f}) page50={json_page:.2f}ms "
        f"retain={json_retain:.2f}ms rows_in_page={len(page)}"
    )
    shutil.rmtree(d)

    # --- sqlite: one INSERT, an index, a LIMIT and a DELETE
    d = _tmpdir()
    store2 = SqliteStore(d / "s.db")
    conn = store2.conn
    latencies = []
    started = time.perf_counter()
    for i in range(total):
        t0 = time.perf_counter()
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO events(correlation_id, kind, body) VALUES (?, ?, ?)",
            (f"c{i % 50}", "run", PAD[:200]),
        )
        conn.execute("COMMIT")
        latencies.append((time.perf_counter() - t0) * 1000)
    sql_wall = time.perf_counter() - started
    sql_bytes = sum(p.stat().st_size for p in d.glob("s.db*"))
    sql_first = statistics.mean(latencies[:100])
    sql_last = statistics.mean(latencies[-100:])
    t0 = time.perf_counter()
    page2 = conn.execute(
        "SELECT seq, body FROM events ORDER BY seq LIMIT 50 OFFSET 4000"
    ).fetchall()
    sql_page = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("DELETE FROM events WHERE seq <= (SELECT max(seq) - 1000 FROM events)")
    conn.execute("COMMIT")
    sql_retain = (time.perf_counter() - t0) * 1000
    print(
        f"  sqlite       wall={sql_wall:.2f}s bytes={sql_bytes:,} "
        f"append_first100={sql_first:.2f}ms append_last100={sql_last:.2f}ms "
        f"(x{sql_last / sql_first:.1f}) page50={sql_page:.2f}ms "
        f"retain={sql_retain:.2f}ms rows_in_page={len(page2)}"
    )

    # idempotent delivery: the same key twice is a constraint violation, not a
    # read-then-write race.
    dup = 0
    for _ in range(2):
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO deliveries(channel, idempotency_key) VALUES (?, ?)",
                ("telegram", "evt-42"),
            )
            conn.execute("COMMIT")
        except sqlite3.IntegrityError:
            conn.execute("ROLLBACK")
            dup += 1
    print(f"  sqlite       duplicate delivery rejected by UNIQUE: {dup}/1")
    shutil.rmtree(d)


# --------------------------------------------------------------------------
# 4. crash injection
# --------------------------------------------------------------------------


def _crash_child(kind: str, path: Path, ready: int) -> None:
    """Write forever; the parent SIGKILLs us mid-flight."""
    store: Any = JsonStore(path) if kind == "json" else SqliteStore(path)
    os.write(ready, b"x")
    i = 0
    while True:
        store.create(f"k{i}", {"i": i, "pad": PAD})
        i += 1


def scenario_crash(trials: int = 8) -> None:
    _section(f"4 crash: SIGKILL a writer mid-flight, {trials} trials each")
    for kind, fname in (("json", "s.json"), ("sqlite", "s.db")):
        readable = 0
        unreadable: list[str] = []
        counts: list[int] = []
        for _ in range(trials):
            d = _tmpdir()
            path = d / fname
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r)
                try:
                    _crash_child(kind, path, w)
                finally:
                    os._exit(0)
            os.close(w)
            os.read(r, 1)
            os.close(r)
            time.sleep(0.05)
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            try:
                store: Any = JsonStore(path) if kind == "json" else SqliteStore(path)
                counts.append(store.count())
                readable += 1
            except Exception as exc:  # noqa: BLE001
                unreadable.append(f"{type(exc).__name__}: {exc}")
            shutil.rmtree(d)
        print(
            f"  {kind:<12} readable_after_kill={readable}/{trials} "
            f"records={counts} leftovers={unreadable or 'none'}"
        )


# --------------------------------------------------------------------------
# 5. two processes writing at once
# --------------------------------------------------------------------------


def _proc_child(kind: str, path: Path, tag: str, rounds: int) -> int:
    store: Any = JsonStore(path) if kind == "json" else SqliteStore(path)
    errs = 0
    for i in range(rounds):
        try:
            store.create(f"{tag}-{i}", {"pad": PAD})
        except Exception:  # noqa: BLE001
            errs += 1
    return errs


def scenario_procs(rounds: int = 150) -> None:
    _section(f"5 procs: 2 PROCESSES x {rounds} writes into one store")
    for kind, fname in (("json", "s.json"), ("sqlite", "s.db")):
        d = _tmpdir()
        path = d / fname
        (JsonStore(path) if kind == "json" else SqliteStore(path))
        pids = []
        pipes = []
        for tag in ("a", "b"):
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r)
                code = 0
                try:
                    code = _proc_child(kind, path, tag, rounds)
                finally:
                    os.write(w, str(min(code, 255)).encode().ljust(4))
                    os._exit(0)
            os.close(w)
            pids.append(pid)
            pipes.append(r)
        errs = 0
        for r in pipes:
            errs += int(os.read(r, 4).decode().strip() or 0)
            os.close(r)
        for pid in pids:
            os.waitpid(pid, 0)
        try:
            final = (JsonStore(path) if kind == "json" else SqliteStore(path)).count()
            verdict = f"on_disk={final}/{rounds * 2}"
        except Exception as exc:  # noqa: BLE001
            verdict = f"UNREADABLE {type(exc).__name__}: {exc}"
        print(f"  {kind:<12} raised={errs} {verdict}")
        shutil.rmtree(d)


# --------------------------------------------------------------------------
# 6. legacy JSON migration
# --------------------------------------------------------------------------


def _migrate(legacy: Path, db: Path) -> str:
    """One store's import: version-gated, backed up, idempotent, loud on damage."""
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version >= 1:
        return "skipped (already at user_version=1)"
    if not legacy.exists():
        conn.execute("PRAGMA user_version=1")
        return "no legacy file; marked migrated"
    try:
        payload = json.loads(legacy.read_text())
    except json.JSONDecodeError as exc:
        return f"REFUSED: {legacy.name} is damaged at line {exc.lineno} col {exc.colno}: {exc.msg}"
    backup = legacy.with_suffix(legacy.suffix + ".pre-sqlite.bak")
    shutil.copy2(legacy, backup)
    conn.execute("BEGIN IMMEDIATE")
    try:
        for key, row in payload.items():
            conn.execute(
                "INSERT INTO rows(key, body) VALUES (?, ?)", (key, json.dumps(row))
            )
        conn.execute("PRAGMA user_version=1")
        conn.execute("COMMIT")
    except Exception as exc:  # noqa: BLE001
        conn.execute("ROLLBACK")
        return f"ROLLED BACK, legacy untouched: {type(exc).__name__}: {exc}"
    return f"imported {len(payload)} row(s), backup at {backup.name}"


def scenario_migrate() -> None:
    _section("6 migrate: legacy JSON -> sqlite, idempotency / backup / damage")
    d = _tmpdir()
    legacy = d / "projects.json"
    legacy.write_text(json.dumps({f"p{i}": {"name": f"proj{i}"} for i in range(25)}))
    db = d / "state.db"
    print(f"  first run   : {_migrate(legacy, db)}")
    print(f"  second run  : {_migrate(legacy, db)}")
    n = sqlite3.connect(db).execute("SELECT count(*) FROM rows").fetchone()[0]
    print(f"  rows after two runs: {n} (25 means the import did not double)")
    print(f"  backup kept : {(legacy.with_suffix('.json.pre-sqlite.bak')).exists()}")

    d2 = _tmpdir()
    damaged = d2 / "projects.json"
    damaged.write_text('{"p0": {"name": "ok"}}\n{"trailing": "garbage"}')
    print(f"  damaged file: {_migrate(damaged, d2 / 'state.db')}")
    print(f"  damaged legacy still present: {damaged.exists()}")
    shutil.rmtree(d)
    shutil.rmtree(d2)


# --------------------------------------------------------------------------
# 7. pytest isolation cost
# --------------------------------------------------------------------------


def scenario_isolation(n: int = 200) -> None:
    _section(f"7 isolation: cost of a fresh empty store, {n} times (one per test)")
    d = _tmpdir()
    t0 = time.perf_counter()
    for i in range(n):
        JsonStore(d / f"j{i}.json").create("seed", {"ok": True})
    json_ms = (time.perf_counter() - t0) / n * 1000
    t0 = time.perf_counter()
    for i in range(n):
        SqliteStore(d / f"s{i}.db").create("seed", {"ok": True})
    file_ms = (time.perf_counter() - t0) / n * 1000
    t0 = time.perf_counter()
    for _ in range(n):
        conn = sqlite3.connect(":memory:", isolation_level=None)
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO rows(key, body) VALUES ('seed', '{}')")
        conn.close()
    mem_ms = (time.perf_counter() - t0) / n * 1000
    print(f"  locked JSON      {json_ms:.3f}ms per store")
    print(f"  sqlite file      {file_ms:.3f}ms per store (tmp_path, synchronous=FULL)")
    print(f"  sqlite :memory:  {mem_ms:.3f}ms per store")
    # Does relaxing durability buy enough to be worth diverging from production?
    for sync in ("FULL", "NORMAL", "OFF"):
        t0 = time.perf_counter()
        for i in range(n):
            conn = sqlite3.connect(d / f"{sync}{i}.db", isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA synchronous={sync}")
            conn.executescript(SCHEMA)
            conn.execute("INSERT INTO rows(key, body) VALUES ('seed', '{}')")
            conn.close()
        print(
            f"    synchronous={sync:<7} {(time.perf_counter() - t0) / n * 1000:.3f}ms"
        )
    shutil.rmtree(d)


# --------------------------------------------------------------------------
# 8. read latency while writers hammer (the "does it block the loop" axis)
# --------------------------------------------------------------------------


def scenario_readers(seed: int = 3000, samples: int = 300) -> None:
    _section(
        f"8 readers: read latency while {WRITERS} writers hammer a {seed}-row store"
    )
    for kind, fname in (("locked JSON", "s.json"), ("sqlite", "s.db")):
        d = _tmpdir()
        path = d / fname
        store: Any = JsonStore(path) if kind == "locked JSON" else SqliteStore(path)
        for i in range(seed):
            store.create(f"seed{i}", {"pad": PAD[:200]})
        stop = threading.Event()
        done = [0]
        wlock = threading.Lock()

        def writer(
            tag: int,
            store: Any = store,
            stop: threading.Event = stop,
            done: list[int] = done,
            wlock: threading.Lock = wlock,
        ) -> None:
            i = 0
            while not stop.is_set():
                store.create(f"w{tag}-{i}", {"pad": PAD[:200]})
                i += 1
                with wlock:
                    done[0] += 1

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(WRITERS)]
        for t in threads:
            t.start()
        lat: list[float] = []
        for _ in range(samples):
            # `count` is the cheapest possible read. The JSON reader must take
            # the same lock the writers hold, or it can observe a row set
            # mid-mutation; that wait IS the number this scenario reports.
            t0 = time.perf_counter()
            store.count()
            lat.append((time.perf_counter() - t0) * 1000)
        stop.set()
        for t in threads:
            t.join()
        lat.sort()
        print(
            f"  {kind:<12} reads p50={lat[len(lat) // 2]:.2f}ms "
            f"p99={lat[int(len(lat) * 0.99)]:.2f}ms max={lat[-1]:.2f}ms "
            f"(writes completed during sampling: {done[0]})"
        )
        shutil.rmtree(d)


# --------------------------------------------------------------------------
# 9. what a SIGKILL leaves behind on disk
# --------------------------------------------------------------------------


def scenario_leftovers(trials: int = 6) -> None:
    _section(f"9 leftovers: files left in the state dir after {trials} SIGKILLs")
    for kind, fname in (("json", "s.json"), ("sqlite", "s.db")):
        stray: list[str] = []
        for _ in range(trials):
            d = _tmpdir()
            path = d / fname
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r)
                try:
                    _crash_child(kind, path, w)
                finally:
                    os._exit(0)
            os.close(w)
            os.read(r, 1)
            os.close(r)
            time.sleep(0.05)
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            stray.extend(p.name for p in d.iterdir() if p.name != fname)
            shutil.rmtree(d)
        kinds = sorted({name.split(".", 1)[1] for name in stray})
        print(f"  {kind:<12} stray files={len(stray)} suffixes={kinds or 'none'}")


# --------------------------------------------------------------------------
# 10. event-loop lag caused by committing from the loop thread
# --------------------------------------------------------------------------


async def _measure_lag(
    commit: Callable[[int], Any], n: int, *, offload: bool
) -> list[float]:
    """Sample how late a 5ms heartbeat runs while `n` commits are issued."""
    lag: list[float] = []
    stop = False

    async def heartbeat() -> None:
        while not stop:
            t0 = time.perf_counter()
            await asyncio.sleep(0.005)
            lag.append((time.perf_counter() - t0 - 0.005) * 1000)

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.02)
    for i in range(n):
        if offload:
            await asyncio.to_thread(commit, i)
        else:
            commit(i)
        await asyncio.sleep(0)
    stop = True
    await beat
    lag.sort()
    return lag


def scenario_asyncio(seed: int = 3000, commits: int = 200) -> None:
    _section(f"10 asyncio: loop lag while committing {commits} times, {seed}-row store")
    for kind, offload in (
        ("locked JSON", False),
        ("locked JSON", True),
        ("sqlite", False),
        ("sqlite", True),
    ):
        d = _tmpdir()
        path = d / ("s.json" if kind == "locked JSON" else "s.db")
        store: Any = JsonStore(path) if kind == "locked JSON" else SqliteStore(path)
        for i in range(seed):
            store.create(f"seed{i}", {"pad": PAD[:200]})

        def one_commit(i: int, store: Any = store) -> None:
            store.create(f"c{i}", {"pad": PAD[:200]})

        lag = asyncio.run(_measure_lag(one_commit, commits, offload=offload))
        where = "to_thread" if offload else "on the loop thread"
        print(
            f"  {kind:<12} {where:<17} lag p50={lag[len(lag) // 2]:.2f}ms "
            f"p99={lag[int(len(lag) * 0.99)]:.2f}ms max={lag[-1]:.2f}ms samples={len(lag)}"
        )
        shutil.rmtree(d)


SCENARIOS: dict[str, Callable[[], None]] = {
    "race": scenario_race,
    "multi": scenario_multi,
    "events": scenario_events,
    "crash": scenario_crash,
    "procs": scenario_procs,
    "migrate": scenario_migrate,
    "isolation": scenario_isolation,
    "readers": scenario_readers,
    "leftovers": scenario_leftovers,
    "asyncio": scenario_asyncio,
}


def main(argv: list[str]) -> int:
    wanted = argv[1:] or list(SCENARIOS)
    unknown = [name for name in wanted if name not in SCENARIOS]
    if unknown:
        print(f"unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(SCENARIOS)}", file=sys.stderr)
        return 2
    print(
        f"python {sys.version.split()[0]}  sqlite {sqlite3.sqlite_version}  cpus {os.cpu_count()}"
    )
    for name in wanted:
        SCENARIOS[name]()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
