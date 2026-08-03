"""The file-size guard is a ratchet, so its own tests must pin both directions.

Like `tests/test_release.py`, the last test here runs against the REAL tree:
if a file grows past its cap without an allowlist entry, or an allowlisted file
shrinks back under its cap and the entry is left behind, this fails - and so
does `nix flake check`.

The edge cases run against a fixture tree, because the point of an edge case is
a repository state we do not want to be in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_file_size
from scripts.check_file_size import (
    ALLOWLIST,
    SOURCE_CAP,
    TEST_CAP,
    cap_for,
    check,
    covered_files,
    main,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write(root: Path, relative: str, lines: int) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n" * lines, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def test_cap_for_assigns_source_and_test_caps() -> None:
    assert cap_for("scufris/app.py") == SOURCE_CAP
    assert cap_for("scufris/hostd/actions.py") == SOURCE_CAP
    assert cap_for("web/src/common.ts") == SOURCE_CAP
    assert cap_for("tests/test_app.py") == TEST_CAP
    assert cap_for("web/src/host-view.test.ts") == TEST_CAP


def test_cap_for_skips_uncovered_extensions() -> None:
    """`.css`, `.html`, and `.json` carry no cap.

    `web/src/style.css` is the largest file in the tree after `tests/test_app.py`
    and no task owns a split for it, so covering it would make the allowlist a
    permanent config knob instead of a ratchet.
    """
    assert cap_for("web/src/style.css") is None
    assert cap_for("web/src/index.html") is None
    assert cap_for("package.json") is None
    assert cap_for("scripts/release_tools.py") is None


def test_covered_files_skips_vendored_and_build_trees(tmp_path: Path) -> None:
    _write(tmp_path, "scufris/app.py", 1)
    _write(tmp_path, "scufris/__pycache__/app.py", 1)
    _write(tmp_path, "web/src/common.ts", 1)
    _write(tmp_path, "web/src/node_modules/dep/index.ts", 1)
    _write(tmp_path, "tests/result-1/thing.py", 1)

    assert covered_files(tmp_path) == ["scufris/app.py", "web/src/common.ts"]


def test_covered_files_keeps_a_source_file_named_like_a_build_output(
    tmp_path: Path,
) -> None:
    """The skip rule is about directories.

    A view named `result-view.ts` is ordinary source; matching the rule against
    its BASENAME would exempt it from the cap silently, which is the one thing
    the guard must never do.
    """
    _write(tmp_path, "web/src/result-view.ts", SOURCE_CAP + 1)

    assert covered_files(tmp_path) == ["web/src/result-view.ts"]
    assert check(tmp_path, frozenset()) != []


# --------------------------------------------------------------------------
# The cap
# --------------------------------------------------------------------------


def test_check_file_size_flags_oversized_file(tmp_path: Path) -> None:
    _write(tmp_path, "scufris/app.py", SOURCE_CAP + 1)

    problems = check(tmp_path, frozenset())

    assert len(problems) == 1
    assert "scufris/app.py" in problems[0]
    assert str(SOURCE_CAP + 1) in problems[0]
    assert str(SOURCE_CAP) in problems[0]


def test_check_file_size_accepts_file_at_the_cap(tmp_path: Path) -> None:
    _write(tmp_path, "scufris/app.py", SOURCE_CAP)
    _write(tmp_path, "tests/test_app.py", TEST_CAP)

    assert check(tmp_path, frozenset()) == []


def test_check_file_size_applies_the_test_cap_to_test_files(tmp_path: Path) -> None:
    """A test file between the two caps is fine; past the test cap it is not."""
    _write(tmp_path, "tests/test_app.py", SOURCE_CAP + 1)
    _write(tmp_path, "web/src/host-view.test.ts", TEST_CAP + 1)

    problems = check(tmp_path, frozenset())

    assert len(problems) == 1
    assert "web/src/host-view.test.ts" in problems[0]


# --------------------------------------------------------------------------
# The ratchet
# --------------------------------------------------------------------------


def test_check_file_size_accepts_allowlisted_oversized_file(tmp_path: Path) -> None:
    _write(tmp_path, "scufris/app.py", SOURCE_CAP + 1)

    assert check(tmp_path, frozenset({"scufris/app.py"})) == []


def test_check_file_size_rejects_stale_allowlist_entry(tmp_path: Path) -> None:
    """An allowlisted file back under its cap must lose its entry.

    This is the whole ratchet: without it, an entry survives the split that was
    supposed to retire it and the file is free to grow again.
    """
    _write(tmp_path, "scufris/app.py", SOURCE_CAP)

    problems = check(tmp_path, frozenset({"scufris/app.py"}))

    assert len(problems) == 1
    assert "scufris/app.py" in problems[0]
    assert "stale" in problems[0]


def test_check_file_size_rejects_allowlist_entry_for_missing_file(
    tmp_path: Path,
) -> None:
    """A deleted or renamed file leaves a stale entry too."""
    problems = check(tmp_path, frozenset({"scufris/gone.py"}))

    assert len(problems) == 1
    assert "scufris/gone.py" in problems[0]
    assert "stale" in problems[0]


def test_check_file_size_reports_every_offender(tmp_path: Path) -> None:
    """One run names every problem; a guard that stops at the first costs a
    round trip per offender."""
    _write(tmp_path, "scufris/app.py", SOURCE_CAP + 1)
    _write(tmp_path, "scufris/agent.py", SOURCE_CAP + 1)
    _write(tmp_path, "tests/test_app.py", SOURCE_CAP)

    problems = check(tmp_path, frozenset({"tests/test_app.py"}))

    assert len(problems) == 3


# --------------------------------------------------------------------------
# The live tree
# --------------------------------------------------------------------------


def test_check_file_size_passes_on_the_repository() -> None:
    assert check(REPO_ROOT, ALLOWLIST) == []


def test_allowlist_holds_repo_relative_posix_paths() -> None:
    for entry in ALLOWLIST:
        assert not entry.startswith("/")
        assert "\\" not in entry
        assert cap_for(entry) is not None


def test_main_exits_zero_on_the_repository(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main() == 0
    assert capsys.readouterr().err == ""


def test_main_exits_nonzero_and_names_every_offender(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path, "scufris/app.py", SOURCE_CAP + 1)
    _write(tmp_path, "scufris/agent.py", SOURCE_CAP + 1)
    monkeypatch.setattr(check_file_size, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check_file_size, "ALLOWLIST", frozenset())

    assert main() == 1

    stderr = capsys.readouterr().err
    assert "scufris/app.py" in stderr
    assert "scufris/agent.py" in stderr
