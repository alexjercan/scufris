"""Fail when a source or test file grows past its line cap.

Context is the scarce resource: a 3800-line module cannot be read, changed, and
re-read inside one working budget, so it is read partially and changed wrongly.
The cap makes that cost visible at the gate instead of at the fifth grep.

Caps: 600 lines for source, 900 for tests. Tests get more room because a test
file is a list of independent cases, not a control-flow surface.

Covered: `scufris/**/*.py`, `tests/**/*.py`, and `web/src/**/*.ts` (with
`*.test.ts` taking the test cap). Stylesheets, HTML, and JSON are not covered.

`ALLOWLIST` is a RATCHET, not a config knob. Every entry is a file some task
already owns and will split; that task deletes its entries in the same change.
Three rules make it one-way:

- over the cap and allowlisted -> pass
- over the cap and not allowlisted -> fail (no new entries)
- inside the cap and allowlisted -> fail as a stale entry (entries must leave)

Run it directly, or via `nix flake check` (the `filesize` check):

    python scripts/check_file_size.py
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Repo root, relative to this file (scripts/check_file_size.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent

SOURCE_CAP = 600
TEST_CAP = 900

#: Trees that are covered. Anything outside them carries no cap.
COVERED_ROOTS = ("scufris", "tests", "web/src")

#: Vendored, generated, or build-output DIRECTORIES, matched by name at any
#: depth. `result` and `result-*` are `nix build` outputs. Matched against
#: directory components only: a source file may legitimately be named
#: `result-view.ts`, and skipping it by basename would exempt it from the cap.
SKIP_DIRS = frozenset({"__pycache__", "node_modules", ".venv"})

#: Files over their cap today. Each is owned by a task that splits it and
#: removes the entry; see tasks/ for the epic. Do not add to this list - a new
#: oversized file is the failure the guard exists to report.
ALLOWLIST: frozenset[str] = frozenset(
    {
        "scufris/app.py",
        "tests/test_app.py",
    }
)


def cap_for(relative: str) -> int | None:
    """The line cap for a repo-relative POSIX path, or None if uncovered."""
    if not any(
        relative == root or relative.startswith(f"{root}/") for root in COVERED_ROOTS
    ):
        return None
    if relative.startswith("tests/") and relative.endswith(".py"):
        return TEST_CAP
    if relative.endswith(".test.ts"):
        return TEST_CAP
    if relative.endswith(".py") or relative.endswith(".ts"):
        return SOURCE_CAP
    return None


def _is_skipped(relative: str) -> bool:
    """Is any DIRECTORY on the way to `relative` a vendored or build tree?"""
    return any(
        part in SKIP_DIRS or part == "result" or part.startswith("result-")
        for part in relative.split("/")[:-1]
    )


def covered_files(root: Path) -> list[str]:
    """Every covered file under `root`, as sorted repo-relative POSIX paths.

    Walks the covered roots rather than the whole tree, so `.git` and
    `web/node_modules` are never enumerated.
    """
    found = []
    for name in COVERED_ROOTS:
        for path in (root / name).rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if _is_skipped(relative) or cap_for(relative) is None:
                continue
            found.append(relative)
    return sorted(found)


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def check(root: Path, allowlist: frozenset[str]) -> list[str]:
    """Every cap violation and every stale allowlist entry, as messages.

    Returns an empty list when the tree is clean. Reports ALL problems rather
    than the first, so one run is one round trip.
    """
    problems = []
    covered = covered_files(root)
    for relative in covered:
        cap = cap_for(relative)
        assert cap is not None  # covered_files filters uncovered paths out
        lines = line_count(root / relative)
        if lines > cap and relative not in allowlist:
            problems.append(f"{relative}: {lines} lines, cap {cap}")
        elif lines <= cap and relative in allowlist:
            problems.append(
                f"{relative}: {lines} lines, cap {cap} - stale allowlist entry, "
                "remove it from ALLOWLIST in scripts/check_file_size.py"
            )
    for relative in sorted(allowlist - set(covered)):
        problems.append(
            f"{relative}: no such covered file - stale allowlist entry, remove "
            "it from ALLOWLIST in scripts/check_file_size.py"
        )
    return problems


def main() -> int:
    problems = check(REPO_ROOT, ALLOWLIST)
    if not problems:
        return 0
    print("File size guard failed:", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    print(
        f"\nCaps: {SOURCE_CAP} lines for source, {TEST_CAP} for tests. "
        "Split the file; do not add an allowlist entry.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
