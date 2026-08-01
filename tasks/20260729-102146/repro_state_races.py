#!/usr/bin/env python3
"""Reproduce the concurrent-write failures of the app-owned JSON stores.

Evidence for task 20260729-102146. Runs the REAL store classes
(``scufris.projects.ProjectStore``, ``scufris.agent_store.AgentStore``) against a
throwaway state dir, from several OS threads at once - the same shape FastAPI
produces when it dispatches a synchronous ``def`` route into the anyio worker
thread pool.

Two failures are demonstrated, both consequences of one write discipline:

    tmp = self._path.with_suffix(".json.tmp")   # FIXED, shared by every writer
    tmp.write_text(json.dumps(payload))         # not atomic, multiple write(2)
    os.replace(tmp, self._path)                 # atomic, but of a shared file

    A. shared-tmp corruption - writer 2 truncates and rewrites the same tmp file
       while writer 1 is still filling it, and either writer's ``os.replace``
       can publish the half-written mixture as the store. The tolerant loader
       then drops the WHOLE file, so one collision loses every record.

    B. lost update - each writer serializes a full snapshot of the shared dict
       and then replaces the file. A writer that serialized early but replaced
       late publishes a snapshot missing records other writers already
       committed. Nothing raises; the on-disk store silently regresses.

Usage (from the repository root):

    python tasks/20260729-102146/repro_state_races.py
    python tasks/20260729-102146/repro_state_races.py --writers 16 --rounds 40

Exit code 0 means at least one failure was reproduced (the point of the
script); 1 means the run was clean and the numbers below should be raised.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scufris.config import Settings  # noqa: E402
from scufris.enums import AgentState  # noqa: E402
from scufris.projects import ProjectStore  # noqa: E402


def _settings(state_dir: Path) -> Settings:
    """A Settings pointing at a throwaway state dir, with writes enabled."""
    return Settings(state_dir=state_dir, settings_writable=True)


# A description long enough that one snapshot spans many write(2) calls: the
# text layer flushes in 8 KiB chunks, so a small store would be written in a
# single syscall and the interleave would be much rarer than it is in
# production, where the file grows with the record count.
_PADDING = "x" * 4096


def _observe(state_dir: Path, name: str) -> tuple[int, str]:
    """What a fresh process would read back: (records, verdict)."""
    path = state_dir / name
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except ValueError as exc:
        return 0, f"CORRUPT ({exc}); {len(raw)} bytes on disk"
    if not isinstance(data, list | dict):
        return 0, f"unexpected top-level {type(data).__name__}"
    return len(data), "parses"


def scenario_projects(state_dir: Path, *, writers: int, rounds: int) -> dict[str, object]:
    """Concurrent ``POST /api/projects``: N worker threads, one shared store."""
    cwd = state_dir / "workspace"
    cwd.mkdir(parents=True, exist_ok=True)
    store = ProjectStore(_settings(state_dir))
    errors: list[str] = []
    start = threading.Barrier(writers)

    def create_many(worker: int) -> None:
        start.wait()
        for n in range(rounds):
            try:
                store.create(
                    f"proj {worker} {n}",
                    str(cwd),
                    description=_PADDING,
                )
            except Exception:  # noqa: BLE001 - the failure IS the observation
                errors.append(traceback.format_exc())

    with ThreadPoolExecutor(max_workers=writers) as pool:
        list(pool.map(create_many, range(writers)))

    expected = writers * rounds
    in_memory = len(store.list())
    on_disk, verdict = _observe(state_dir, "projects.json")
    # What a restart would actually recover.
    reloaded = len(ProjectStore(_settings(state_dir)).list())
    return {
        "store": "projects.json",
        "expected": expected,
        "in_memory": in_memory,
        "on_disk": on_disk,
        "after_restart": reloaded,
        "file_verdict": verdict,
        "exceptions": errors,
    }


def scenario_agents(state_dir: Path, *, writers: int, rounds: int) -> dict[str, object]:
    """A CRUD route racing the run engine: ``create`` vs ``mark_finished``.

    ``create``/``delete`` run in the anyio worker thread pool (synchronous route
    handlers); ``mark_finished`` runs from the supervisor's completion callback
    on the event loop thread. Both call ``AgentStore._persist`` on agents.json,
    and ``mark_finished`` additionally writes outcomes.json and sessions.json.
    """
    from scufris.agent_store import AgentStore

    cwd = state_dir / "workspace"
    cwd.mkdir(parents=True, exist_ok=True)
    projects = ProjectStore(_settings(state_dir))
    project = projects.create("agents workspace", str(cwd))
    store = AgentStore(_settings(state_dir), projects)
    errors: list[str] = []
    start = threading.Barrier(writers)

    def churn(worker: int) -> None:
        start.wait()
        for n in range(rounds):
            try:
                agent = store.create(
                    f"agent {worker} {n}", project.id, description=_PADDING
                )
                store.mark_finished(
                    agent.id,
                    state=AgentState.DONE,
                    session_id=f"sess-{worker}-{n}",
                    message=_PADDING,
                    run_id=f"run-{worker}-{n}",
                )
            except Exception:  # noqa: BLE001
                errors.append(traceback.format_exc())

    with ThreadPoolExecutor(max_workers=writers) as pool:
        list(pool.map(churn, range(writers)))

    expected = writers * rounds
    in_memory = len(store._agents)  # noqa: SLF001 - reading the truth, not the API
    results = []
    for name in ("agents.json", "outcomes.json", "sessions.json"):
        on_disk, verdict = _observe(state_dir, name)
        results.append({"file": name, "on_disk": on_disk, "verdict": verdict})
    reloaded = AgentStore(_settings(state_dir), projects)
    return {
        "store": "agents.json + outcomes.json + sessions.json",
        "expected": expected,
        "in_memory": in_memory,
        "files": results,
        "after_restart": len(reloaded._agents),  # noqa: SLF001
        "outcomes_after_restart": len(reloaded.outcomes()),
        "exceptions": errors,
    }


class _UniqueTmpStore:
    """The same discipline with the ONE obvious fix: a per-writer tmp name.

    This is not production code; it exists to answer "is the shared tmp path the
    whole bug?". It keeps everything else the stores do - a shared in-memory
    dict, a full snapshot serialized per write, ``os.replace`` to publish - and
    only makes the temporary file unique. Any record still missing from the
    published file after this run is a lost update that renaming the tmp file
    does not fix.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: dict[str, str] = {}

    def add(self, key: str, value: str) -> None:
        self._records[key] = value
        self._persist()

    def _persist(self) -> None:
        payload = json.dumps(self._records)
        tmp = self._path.with_name(f"{self._path.name}.{threading.get_ident()}.tmp")
        tmp.write_text(payload)
        os.replace(tmp, self._path)


def scenario_lost_update(
    state_dir: Path, *, writers: int, rounds: int
) -> dict[str, object]:
    """The window a unique tmp name leaves open: a stale snapshot published late."""
    path = state_dir / "unique-tmp.json"
    path.write_text("{}")
    store = _UniqueTmpStore(path)
    errors: list[str] = []
    start = threading.Barrier(writers + 1)
    done = threading.Event()
    # The published file is sampled continuously. A count that goes DOWN is a
    # writer publishing a snapshot older than one already on disk: crash, kill or
    # power-cut at that instant and those records are gone. The final file is not
    # evidence either way - the writers share one in-memory dict, so whoever
    # replaces LAST happens to publish everything.
    regressions: list[tuple[int, int]] = []

    def watch() -> None:
        start.wait()
        high = 0
        while not done.is_set():
            try:
                seen = len(json.loads(path.read_text()))
            except (OSError, ValueError):
                continue
            if seen < high:
                regressions.append((high, seen))
            high = max(high, seen)

    def add_many(worker: int) -> None:
        start.wait()
        for n in range(rounds):
            try:
                store.add(f"{worker}-{n}", _PADDING)
            except Exception:  # noqa: BLE001
                errors.append(traceback.format_exc())

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    with ThreadPoolExecutor(max_workers=writers) as pool:
        list(pool.map(add_many, range(writers)))
    done.set()
    watcher.join(timeout=5)

    on_disk, verdict = _observe(state_dir, "unique-tmp.json")
    worst = max((high - seen for high, seen in regressions), default=0)
    return {
        "store": "unique tmp name (control: is the shared tmp path the whole bug?)",
        "expected": writers * rounds,
        "in_memory": len(store._records),  # noqa: SLF001
        "on_disk": on_disk,
        "after_restart": on_disk,
        "file_verdict": verdict,
        "published_regressions": len(regressions),
        "worst_regression_records": worst,
        "exceptions": errors,
    }


def scenario_reasoning(state_dir: Path, *, writers: int, rounds: int) -> dict[str, object]:
    """The same collision where the store SWALLOWS it: the reasoning sidecar.

    ``ReasoningStore._persist`` catches ``OSError``, and the ``os.replace``
    failure above IS an ``OSError``. One session's turns are appended from
    whichever run streams them, so two turns of one session finishing together
    lose reasoning with no exception, no error log the caller sees, and no
    difference in the API response.
    """
    from scufris.reasoning_store import ReasoningStore

    store = ReasoningStore(_settings(state_dir))
    session = "shared-session"
    start = threading.Barrier(writers)

    def append_many(worker: int) -> None:
        start.wait()
        for n in range(rounds):
            store.append(session, _PADDING, answer=f"answer {worker} {n}")

    with ThreadPoolExecutor(max_workers=writers) as pool:
        list(pool.map(append_many, range(writers)))

    kept = len(ReasoningStore(_settings(state_dir)).read(session))
    return {
        "store": "reasoning/<session>.json (errors swallowed by design)",
        "expected": writers * rounds,
        "in_memory": kept,
        "on_disk": kept,
        "after_restart": kept,
        "file_verdict": "parses" if kept else "empty or unreadable",
        "exceptions": [],
    }


def _report(result: dict[str, object]) -> bool:
    """Print one scenario; return True if it reproduced a failure."""
    print(f"\n--- {result['store']} ---")
    for key, value in result.items():
        if key in {"store", "exceptions", "files"}:
            continue
        print(f"  {key}: {value}")
    for row in result.get("files", []):  # type: ignore[union-attr]
        print(f"  {row['file']}: {row['on_disk']} record(s), {row['verdict']}")
    errors = result["exceptions"]
    assert isinstance(errors, list)
    print(f"  exceptions raised: {len(errors)}")
    if errors:
        print("  first traceback:")
        for line in errors[0].strip().splitlines():
            print(f"    {line}")
    lost = result["expected"] != result["after_restart"]
    corrupt = "CORRUPT" in str(result.get("file_verdict", "")) or any(
        "CORRUPT" in str(row["verdict"]) for row in result.get("files", [])  # type: ignore[union-attr]
    )
    if lost:
        print(
            f"  FAILURE: {int(result['expected']) - int(result['after_restart'])} "
            "record(s) unrecoverable after a restart"
        )
    if corrupt:
        print("  FAILURE: the published file does not parse")
    if errors:
        print("  FAILURE: a write raised")
    return bool(lost or corrupt or errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writers", type=int, default=8, help="concurrent threads")
    parser.add_argument("--rounds", type=int, default=25, help="writes per thread")
    args = parser.parse_args()

    print(
        f"concurrency: {args.writers} threads x {args.rounds} writes "
        f"= {args.writers * args.rounds} records per store"
    )
    failed = False
    for scenario in (scenario_projects, scenario_agents, scenario_lost_update, scenario_reasoning):
        root = Path(tempfile.mkdtemp(prefix="scufris-repro-"))
        try:
            failed |= _report(scenario(root, writers=args.writers, rounds=args.rounds))
        finally:
            shutil.rmtree(root, ignore_errors=True)
    print()
    if failed:
        print("reproduced: the stores lose or corrupt records under concurrent writes")
        return 0
    print("clean run: raise --writers/--rounds and try again")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
