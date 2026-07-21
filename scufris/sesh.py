"""Directory-only project discovery + creation (a `sesh`-style sessionizer, minus
the tmux).

`discover()` scans configurable base dirs ONE level deep and returns candidate
project directories, inferring a language from marker files. `create()` makes a
directory under a base and returns its path - NO tmux, NO subprocess. The Projects
page surfaces these candidates unioned with the registered projects (see
tasks/20260721-112212/SPIKE.md decision 4).
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

# Marker file -> language. Checked in insertion order; the first present marker in
# a directory wins. Kept small and obvious; extend as needed.
MARKERS: dict[str, str] = {
    "pyproject.toml": "python",
    "setup.py": "python",
    "requirements.txt": "python",
    "Cargo.toml": "rust",
    "go.mod": "go",
    "package.json": "node",
    "deno.json": "deno",
    "pom.xml": "java",
    "build.gradle": "java",
    "Gemfile": "ruby",
    "composer.json": "php",
    "mix.exs": "elixir",
    "flake.nix": "nix",
}


class Candidate(BaseModel):
    """A discovered project directory: its absolute path, display name (the
    directory's own name) and an inferred language ("" when unknown)."""

    path: str
    name: str
    language: str = ""


def infer_language(directory: Path) -> str:
    """Guess a directory's language from the first marker file it contains."""
    for marker, language in MARKERS.items():
        if (directory / marker).is_file():
            return language
    return ""


def discover(base_dirs: list[Path]) -> list[Candidate]:
    """Scan each base dir ONE level deep for candidate project directories.

    Only immediate subdirectories are returned (not the base dirs themselves and
    not deeper); hidden directories (dotfiles) and non-directories are skipped. A
    missing/ unreadable base dir is ignored. Results are deduped by resolved path
    (a directory reachable through two bases appears once) and sorted by name then
    path for a stable listing.
    """
    seen: dict[str, Candidate] = {}
    for base in base_dirs:
        root = base.expanduser()
        if not root.is_dir():
            continue
        try:
            entries = sorted(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            resolved = str(entry.resolve())
            if resolved in seen:
                continue
            seen[resolved] = Candidate(
                path=resolved,
                name=entry.name,
                language=infer_language(entry),
            )
    return sorted(seen.values(), key=lambda c: (c.name.lower(), c.path))


# A directory name is a single path segment: no separators, no traversal, no
# absolute-path escape. Everything else is a plain name we mkdir under the base.
_SAFE_NAME_RE = re.compile(r"^[^/\\\x00]+$")


def create(name: str, base: Path) -> Path:
    """Make `base/<name>` and return its path - NO tmux, NO subprocess.

    Idempotent: an already-existing directory is fine (registering a discovered
    dir). The name must be a single, non-traversing path segment (no `/`, `\\`,
    NUL, `.` or `..`); anything else raises ValueError so a caller can never mkdir
    outside the base.
    """
    name = name.strip()
    if not name or name in {".", ".."} or not _SAFE_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid project directory name: {name!r}")
    target = base.expanduser() / name
    target.mkdir(parents=True, exist_ok=True)
    return target
