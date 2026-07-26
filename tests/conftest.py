"""Shared test fixtures.

`FakeCollector` returns a deterministic `HostStats`, so backend/API tests never
touch real host state or psutil.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

import scufris as _scufris
from scufris.metrics import (
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
) -> None:
    """Keep every test's persisted stores out of the developer's real
    ``~/.local/state/scufris``.

    ``create_app`` and the stores default ``state_dir`` to the real home, so a
    test that constructs ``Settings()`` without an explicit ``state_dir`` would
    otherwise write agents/sessions/outcomes/reasoning there. The reasoning
    sidecar is append-only, so that would also GROW a real-home file across runs.
    Point the default at a per-test temp dir via the env override; a test that
    passes an explicit ``state_dir`` still wins (init kwargs beat env). See lesson
    isolate-state_dir-in-tests-that-assert-config.
    """
    monkeypatch.setenv("SCUFRIS_STATE_DIR", str(tmp_path_factory.mktemp("state")))


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
