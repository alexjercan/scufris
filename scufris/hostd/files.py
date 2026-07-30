"""The filesystem seam, for the questions a command cannot answer honestly.

``scufris.host.run.Runner`` is the seam for "what did this command say".  R3 also
has to ask "does this store path look like a NixOS system" and "what does this
generation link point at", and those are file reads. Shelling out to ``test`` and
``readlink`` for them would be worse in two ways: the helper's unit PATH holds
``nix``, ``systemd`` and ``nixos-rebuild`` and deliberately not coreutils, and a
security check that fails when a binary is missing is a check with an extra way
to go wrong.

So they go through this seam instead - injectable exactly like the runner and the
executor, so the whole R3 path is exercisable without a NixOS system underneath.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Files(Protocol):
    """The three filesystem questions the privileged verbs ask."""

    def is_file(self, path: str) -> bool:
        """Whether ``path`` exists and is a regular file (symlinks followed)."""
        ...

    def is_executable(self, path: str) -> bool:
        """Whether ``path`` exists and is executable."""
        ...

    def resolve(self, path: str) -> str:
        """What ``path`` points at, fully resolved. Empty when it does not exist.

        Empty rather than raising, and empty rather than the input path: a
        caller must not be able to mistake "could not resolve" for "resolves to
        itself", because for a generation link that would mean activating the
        link instead of the system it names.
        """
        ...


class RealFiles:
    """The real filesystem. The only implementation the helper ships with."""

    def is_file(self, path: str) -> bool:
        try:
            return Path(path).is_file()
        except OSError:
            return False

    def is_executable(self, path: str) -> bool:
        try:
            return os.access(path, os.X_OK) and not Path(path).is_dir()
        except OSError:
            return False

    def resolve(self, path: str) -> str:
        try:
            target = Path(path).resolve(strict=True)
        except (OSError, RuntimeError):
            return ""
        return str(target)


# The real filesystem, as a module-level singleton (it is stateless). Callers
# that have no reason to inject - every R1/R2 path, since only R3 reads files -
# get it by default; the helper's engine always passes its own.
DEFAULT_FILES = RealFiles()


class FakeFiles:
    """A scripted ``Files`` for tests and the example script.

    Lives in the package rather than the tests so ``examples/nixos_change.py``
    can drive a full R3 flow with no NixOS system present.
    """

    def __init__(
        self,
        *,
        files: set[str] | None = None,
        executables: set[str] | None = None,
        links: dict[str, str] | None = None,
    ) -> None:
        self.files = set(files or ())
        self.executables = set(executables or ())
        self.links = dict(links or {})

    def is_file(self, path: str) -> bool:
        return path in self.files

    def is_executable(self, path: str) -> bool:
        return path in self.executables

    def resolve(self, path: str) -> str:
        return self.links.get(path, "")
