"""Tests for directory-only project discovery + creation (no tmux)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from scufris import sesh


def _mkproject(base: Path, name: str, marker: str | None = None) -> Path:
    d = base / name
    d.mkdir(parents=True)
    if marker:
        (d / marker).write_text("")
    return d


def test_discover_finds_one_level_dirs_and_infers_language(tmp_path: Path) -> None:
    base = tmp_path / "personal"
    _mkproject(base, "api", "pyproject.toml")
    _mkproject(base, "web", "package.json")
    _mkproject(base, "cli", "Cargo.toml")
    _mkproject(base, "notes")  # no marker -> unknown language

    cands = sesh.discover([base])

    by_name = {c.name: c for c in cands}
    assert set(by_name) == {"api", "web", "cli", "notes"}
    assert by_name["api"].language == "python"
    assert by_name["web"].language == "node"
    assert by_name["cli"].language == "rust"
    assert by_name["notes"].language == ""
    # Absolute resolved paths, sorted by name.
    assert by_name["api"].path == str((base / "api").resolve())
    assert [c.name for c in cands] == sorted(c.name for c in cands)


def test_discover_skips_files_hidden_dirs_and_goes_only_one_level(
    tmp_path: Path,
) -> None:
    base = tmp_path / "work"
    proj = _mkproject(base, "real")
    _mkproject(proj, "nested", "pyproject.toml")  # one level DEEPER -> ignored
    _mkproject(base, ".hidden")  # dotdir -> ignored
    (base / "loose-file.txt").write_text("x")  # a file -> ignored

    names = {c.name for c in sesh.discover([base])}
    assert names == {"real"}  # not "nested", not ".hidden", not the file


def test_discover_ignores_missing_base_and_dedups_across_bases(
    tmp_path: Path,
) -> None:
    base = tmp_path / "personal"
    _mkproject(base, "api")
    missing = tmp_path / "does-not-exist"
    # The same base listed twice must not double a candidate.
    cands = sesh.discover([base, missing, base])
    assert [c.name for c in cands] == ["api"]


def test_create_makes_the_dir_and_is_idempotent(tmp_path: Path) -> None:
    path = sesh.create("new-thing", tmp_path)
    assert path == tmp_path / "new-thing"
    assert path.is_dir()
    # Idempotent: creating again (e.g. registering a discovered dir) is fine.
    again = sesh.create("new-thing", tmp_path)
    assert again == path and again.is_dir()


def test_create_rejects_traversal_and_separators(tmp_path: Path) -> None:
    for bad in ("../escape", "a/b", "..", ".", "  ", "with/slash", "/etc/foo"):
        with pytest.raises(ValueError):
            sesh.create(bad, tmp_path)
    # None of the rejected names created anything outside the base.
    assert list(tmp_path.iterdir()) == []


def test_sesh_spawns_no_tmux_or_subprocess() -> None:
    """The module is directory-only: it must not shell out (no tmux, no
    subprocess). Guard by CAPABILITY, not prose - assert the module imported no
    process-spawning machinery, so `create` cannot start tmux. (Checking the
    source text would trip on this module's own 'no tmux/subprocess' docstring.)"""
    assert not hasattr(sesh, "subprocess")
    assert not hasattr(sesh, "os")
    assert not hasattr(sesh, "shutil")
    # No spawning call appears in the source's code (docstring words aside).
    code = "\n".join(
        line
        for line in inspect.getsource(sesh).splitlines()
        if not line.strip().startswith("#")
    )
    assert "Popen" not in code and "os.system" not in code
