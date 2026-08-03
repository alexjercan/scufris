"""Shared test fixtures.

`FakeCollector` returns a deterministic `HostStats`, so backend/API tests never
touch real host state or psutil.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import scufris as _scufris
from scufris.auth import CSRF_HEADER, hash_password
from scufris.config import Settings
from scufris.db import (
    Database,
    close_all_state_databases,
    open_database,
    upgrade_to_head,
)
from scufris.enums import AuthPolicy
from scufris.hostd import (
    AuditLog,
    FakeExecutor,
    FakeFiles,
    HostdEngine,
    HostdServer,
)
from scufris_host import (
    DiskUsage,
    HostStats,
    MemStats,
    NetIO,
    SwapStats,
)

# Guard: fail fast if `scufris` resolves from outside the current directory.
# Bare `pytest` (the console script) does not put CWD first on sys.path, so in a
# sprout worktree it imports scufris from the MAIN checkout's editable install
# and silently tests the wrong tree. `python -m pytest` puts CWD first and fixes
# it. See LESSONS.md nix-devshell-import-resolves-to-cwd-source.
# OK when the package root is the cwd or an ancestor of it (running from the
# repo root or any subdirectory of it); fire only when scufris resolves from a
# tree unrelated to cwd (the main checkout, /tmp, etc).
_pkg_root = Path(_scufris.__file__).resolve().parent.parent
_cwd = Path.cwd().resolve()
if _pkg_root != _cwd and _pkg_root not in _cwd.parents:
    raise RuntimeError(
        f"scufris is imported from {_pkg_root}, not the current directory {_cwd}. "
        "Bare `pytest` does not put CWD first on sys.path, so in a sprout worktree "
        "it tests the main checkout. Run `python -m pytest` instead."
    )


@pytest.fixture(autouse=True)
def _isolate_state_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Keep every test's persisted stores out of the developer's real
    ``~/.local/state/scufris``, and keep ``SCUFRIS_*`` env leaks from crossing
    test boundaries.

    ``create_app`` and the stores default ``state_dir`` to the real home, so a
    test that constructs ``Settings()`` without an explicit ``state_dir`` would
    otherwise write agents/sessions/outcomes/reasoning there. The reasoning
    sidecar is append-only, so that would also GROW a real-home file across runs.
    Point the default at a per-test temp dir via the env override; a test that
    passes an explicit ``state_dir`` still wins (init kwargs beat env). See lesson
    isolate-state_dir-in-tests-that-assert-config.

    Env isolation: production bridges like ``ensure_den_path`` /
    ``ensure_api_base`` MUTATE ``os.environ`` directly via ``setdefault`` (to
    hand ``SCUFRIS_DEN_PATH`` / ``SCUFRIS_API_BASE`` to an in-process tool run).
    monkeypatch does NOT track a direct ``os.environ`` write, and ``_env_file=None``
    disables the ``.env`` FILE but not an already-leaked ``os.environ`` var. So on
    a checkout whose ``.env`` sets ``SCUFRIS_DEN_PATH``, an app-creating test would
    seed the var into the process env and a later ``Settings(_env_file=None)`` test
    (e.g. ``test_backends::_hermetic()``) would still read it and wire the ``den``
    server it did not expect. Snapshot every ``SCUFRIS_*`` key before the test and
    restore that exact set after, so no test can leak one into the next. See lesson
    os-environ-setdefault-in-test-leaks-past-monkeypatch.
    """
    monkeypatch.setenv("SCUFRIS_STATE_DIR", str(tmp_path_factory.mktemp("state")))
    # Snapshot AFTER the setenv, so the just-set temp SCUFRIS_STATE_DIR is in the
    # saved set and restore re-asserts it; monkeypatch's own teardown (which runs
    # after this fixture finalizes) is the authority that ultimately removes it.
    saved = snapshot_scufris_env()
    yield
    restore_scufris_env(saved)


@pytest.fixture(autouse=True)
def _close_process_wide_databases() -> Iterator[None]:
    """Drop every handle the process-wide accessor opened during a test.

    `scufris.db.state_database` memoizes one handle per resolved state
    directory, which is what makes `create_app`, an in-process MCP tool and
    `CodexBackend.read_transcript` share ONE database. In a test process that
    memo would otherwise accumulate a handle per temp directory - each holding
    pooled connections to a path pytest has already removed - and a test reusing
    a directory name would be handed a previous test's handle rather than
    opening its own.
    """
    yield
    close_all_state_databases()


def snapshot_scufris_env() -> dict[str, str]:
    """Capture the current ``SCUFRIS_*`` os.environ keys and values."""
    return {k: v for k, v in os.environ.items() if k.startswith("SCUFRIS_")}


def restore_scufris_env(saved: dict[str, str]) -> None:
    """Restore ``os.environ`` to exactly the ``SCUFRIS_*`` set in ``saved``:
    drop keys that appeared since the snapshot, reinstate the snapshot's values."""
    for key in [k for k in os.environ if k.startswith("SCUFRIS_")]:
        if key not in saved:
            del os.environ[key]
    for key, value in saved.items():
        os.environ[key] = value


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "needs_tatr: test shells out to the real tatr CLI; skipped when tatr is "
        "not on PATH (e.g. the nix check sandbox).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip `needs_tatr` tests when the tatr binary is not on PATH.

    The tatr-integration tests run fully in the devShell (tatr on PATH); the
    `nix flake check` sandbox has no tatr, so they are skipped loudly there
    rather than failing the QA gate.
    """
    if shutil.which("tatr") is not None:
        return
    skip = pytest.mark.skip(
        reason="requires the tatr CLI on PATH (absent in the nix check sandbox)"
    )
    for item in items:
        if item.get_closest_marker("needs_tatr"):
            item.add_marker(skip)


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    """A file-backed state database with the production pragmas, AT HEAD.

    File-backed rather than ``:memory:``: a restart-survival proof has to reopen
    the file, and ``:memory:`` has no file to reopen. It costs single-digit
    milliseconds per test, which is affordable.

    Migrated, so a store test gets the schema its store expects rather than an
    empty file. A test that needs a database BEFORE its first migration - only
    `test_db_migrations.py` does - opens its own with ``open_database``.
    """
    db = open_database(tmp_path)
    try:
        upgrade_to_head(db)
        yield db
    finally:
        db.close()


def patch_get_backend(monkeypatch: pytest.MonkeyPatch, backend: Any) -> Any:
    """Route every ``get_backend(...)`` in the serving path to ``backend``.

    ``get_backend`` is imported by name into more than one module, so patching
    the source in ``scufris.backends`` would not rebind the importers. Both
    bind sites are listed here so a test rig has one place to update when a
    caller moves. A LISTED bind site that disappears fails loudly
    (``monkeypatch.setattr`` raises on a missing attribute); a NEW one is
    silent until it is added here, so add it when a module starts importing
    ``get_backend`` - otherwise that caller talks to a real backend while the
    test still looks patched.
    """
    for target in (
        "scufris.api.agent_runs",
        "scufris.api.legacy_agent",
        "scufris.orchestrator.runs",
    ):
        monkeypatch.setattr(f"{target}.get_backend", lambda _name: backend)
    return backend


class FakeCollector:
    def __init__(self, stats: HostStats) -> None:
        self._stats = stats

    def sample(self) -> HostStats:
        return self._stats


def make_fixture_stats() -> HostStats:
    return HostStats(
        hostname="testbox",
        os_name="Linux",
        kernel="6.18.0",
        cpu_percent=12.5,
        per_cpu_percent=[10.0, 15.0],
        mem=MemStats(total=1000, used=400, available=600, percent=40.0),
        swap=SwapStats(total=200, used=50, percent=25.0),
        disks=[DiskUsage(mountpoint="/", total=500, used=100, percent=20.0)],
        load_avg=(0.1, 0.2, 0.3),
        uptime_seconds=1234.0,
        net=NetIO(bytes_sent=10, bytes_recv=20),
        sampled_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def fake_stats() -> HostStats:
    return make_fixture_stats()


@pytest.fixture
def fake_collector(fake_stats: HostStats) -> FakeCollector:
    return FakeCollector(fake_stats)


# The shared secret the helper and the app agree on in tests.
HOSTD_SECRET = "hostd-shared-secret"


# --- a real scufris-hostd, on a real socket ----------------------------------
#
# Shared by every test that drives the privileged path for real: the HTTP approval
# boundary (`test_host_action_api.py`) and the Telegram surface
# (`test_telegram_approvals.py`). It lives here rather than in one of those modules
# because a pytest fixture imported ACROSS test modules is both a ruff F811 and a
# trap - the importing module's parameter shadows the imported name.


class _Helper:
    """A running scufris-hostd, with the knobs a test needs to look inside."""

    def __init__(
        self,
        socket_path: Path,
        executor: FakeExecutor,
        audit: AuditLog,
        runner: Any,
        files: FakeFiles,
        clock: dict[str, float] | None = None,
    ):
        self.socket_path = socket_path
        self.executor = executor
        self.audit = audit
        # The helper's own clock, so a test can let a proposal's approval window
        # close for real (the engine expires on its next read) instead of reaching
        # into its private state.
        self._clock = clock if clock is not None else {"t": 0.0}
        # The faked host itself, so a test can move it: after an activation the
        # generation list and /run/current-system have changed, and a rollback
        # test that pretended otherwise would be testing a machine that cannot
        # exist.
        self.runner = runner
        self.files = files

    def advance(self, seconds: float) -> None:
        """Move the HELPER's clock forward, closing any window that then lapses."""
        self._clock["t"] += seconds


@pytest.fixture
def helper() -> Iterator[_Helper]:
    """Run the real helper on a real unix socket, in its own event loop.

    A short temp directory rather than pytest's ``tmp_path``: a unix socket path
    is capped near 108 bytes and pytest's per-test paths are long enough to
    matter.
    """
    directory = Path(tempfile.mkdtemp(prefix="hostd-"))
    from test_host_actions import host_files, host_runner

    audit = AuditLog(directory / "audit.jsonl", secrets=frozenset({HOSTD_SECRET}))
    executor = FakeExecutor()
    runner = host_runner()
    files = host_files()
    clock = {"t": time.time()}
    engine = HostdEngine(
        audit,
        runner=runner,
        executor=executor,
        files=files,
        clock=lambda: clock["t"],
    )
    socket_path = directory / "h.sock"
    server = HostdServer(engine, secret=HOSTD_SECRET, socket_path=socket_path)

    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True, name="hostd-test")
    thread.start()
    assert ready.wait(timeout=10), "the helper did not start"
    try:
        yield _Helper(socket_path, executor, audit, runner, files, clock)
    finally:
        # Cancel whatever is still in flight BEFORE stopping the loop. A test
        # that cancels a hung apply leaves a connection handler mid-await, and
        # tearing the loop down under it surfaced as a teardown ERROR alongside
        # an unrelated-looking test failure (review round 1, R1.13).
        async def _drain() -> None:
            await server.aclose()
            pending = [
                task
                for task in asyncio.all_tasks(loop)
                if task is not asyncio.current_task()
            ]
            for task in pending:
                task.cancel()
            for task in pending:
                with suppress(asyncio.CancelledError, Exception):
                    await task

        asyncio.run_coroutine_threadsafe(_drain(), loop).result(timeout=15)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()
        shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def make_client() -> Iterator[Callable[[Any], TestClient]]:
    """A TestClient held OPEN for the whole test.

    Entering it as a context manager is what keeps the app's event loop alive
    between requests. Without that, the portal is torn down after each call and
    the supervisor's background apply - which outlives the request by design
    (ADR-001) - is cancelled before it runs. A test that approved and then
    polled would see the action settle with nothing having executed, which is
    an artefact of the harness rather than of the code.
    """
    with ExitStack() as stack:

        def make(app: Any) -> TestClient:
            return stack.enter_context(TestClient(app))

        yield make


# --- driving the app as a logged-in operator ---------------------------------
#
# Shared by every domain that reaches the app through an operator session: the
# host-action boundary, the host digest, the Telegram approval surface and the
# NixOS config-change flow. They live here rather than in one of those modules
# because their consumers are already cross-domain.

PASSWORD = "correct horse battery staple"
ORIGIN = "http://testserver"
SECRET = HOSTD_SECRET


def _settings(tmp_path: Path, helper: _Helper, **kwargs: Any) -> Settings:
    base: dict[str, Any] = {
        "web_dist": tmp_path / "absent",
        "state_dir": tmp_path,
        "auth_mode": AuthPolicy.REQUIRED,
        "auth_password_hash": hash_password(PASSWORD),
        "hostd_socket": helper.socket_path,
        "hostd_secret": SECRET,
        "_env_file": None,
    }
    base.update(kwargs)
    return Settings(**base)


def _login(client: TestClient) -> str:
    resp = client.post(
        "/api/auth/login", json={"password": PASSWORD}, headers={"Origin": ORIGIN}
    )
    assert resp.status_code == 200, resp.text
    return client.cookies["scufris_csrf"]


def _propose(client: TestClient, csrf: str, **body: Any) -> dict[str, Any]:
    payload = {"kind": "unit_restart", "args": {"unit": "nginx"}}
    payload.update(body)
    resp = client.post(
        "/api/host/actions",
        json=payload,
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()
