"""Regression: the autouse ``_isolate_state_dir`` fixture must not let a
production ``os.environ`` bridge leak a ``SCUFRIS_*`` key past a test boundary.

Background (lesson os-environ-setdefault-in-test-leaks-past-monkeypatch): a
production helper like ``ensure_den_path`` calls
``os.environ.setdefault("SCUFRIS_DEN_PATH", ...)`` to hand the den path to an
in-process tool run. monkeypatch does not track that direct write, and
``Settings(_env_file=None)`` disables the ``.env`` FILE but not an already-leaked
``os.environ`` var. On a checkout whose ``.env`` sets ``SCUFRIS_DEN_PATH``, an
app-creating test would seed the var and a later hermetic ``test_backends`` test
would wire a ``den`` server it did not expect. The fix is the fixture's
snapshot/restore of ``SCUFRIS_*`` keys; this test pins that mechanism directly,
without depending on inter-test ordering.
"""

from __future__ import annotations

import os
from pathlib import Path

from conftest import restore_scufris_env, snapshot_scufris_env

from scufris.config import Settings
from scufris.env_bridge import ensure_den_path


def test_ensure_den_path_leak_does_not_survive_snapshot_restore() -> None:
    # Model the fixture boundary: snapshot at "test start" with no den leaked.
    os.environ.pop("SCUFRIS_DEN_PATH", None)
    saved = snapshot_scufris_env()
    assert "SCUFRIS_DEN_PATH" not in saved

    # The production bridge leaks the key into the process env, exactly as an
    # app-creating test triggers via a health / tool-run endpoint.
    ensure_den_path(Settings(den_path=Path("/home/op/the-den"), _env_file=None))  # type: ignore[call-arg]
    assert os.environ["SCUFRIS_DEN_PATH"] == "/home/op/the-den"

    # The fixture's teardown restores the snapshot, so the next test sees a clean
    # env and a hermetic Settings() does not read the leaked den path.
    restore_scufris_env(saved)
    assert "SCUFRIS_DEN_PATH" not in os.environ


def test_restore_reinstates_a_pre_existing_scufris_key() -> None:
    # A key present at snapshot time (e.g. an operator's real exported env) must be
    # RESTORED to its original value, not dropped, even if the test changed it.
    os.environ["SCUFRIS_DEN_PATH"] = "/baseline/den"
    saved = snapshot_scufris_env()

    # A test mutates the key mid-run (as any bridge / setenv would).
    os.environ["SCUFRIS_DEN_PATH"] = "/mutated/den"

    restore_scufris_env(saved)
    assert os.environ["SCUFRIS_DEN_PATH"] == "/baseline/den"
