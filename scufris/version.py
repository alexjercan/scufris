"""The one place the running application learns its own version.

`pyproject.toml` is the source of truth. At runtime we do not parse it - the
installed distribution's metadata carries the same string, put there by the
build, so `importlib.metadata` IS `pyproject.toml` seen from the other side of
packaging. Anything that wants to show a version (the API, the dashboard
footer, the Telegram health card, `scufris --version`) reads it from here, so
there is exactly one fallback string and one place to change.

Before this module there were two copies of this lookup - `app.py` and
`health.py` - with DIFFERENT fallbacks ("0.0.0+unknown" and "unknown"), which
is the disagreement this task exists to end.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

#: What we report when the distribution metadata is missing. That happens only
#: when scufris is imported from a source tree that was never installed (not
#: the dev shell, which installs it editable, and not the Nix package). It is a
#: PEP 440 local version so it can never be mistaken for a real release.
UNKNOWN_VERSION = "0.0.0+unknown"


def scufris_version() -> str:
    """The installed distribution's version, or `UNKNOWN_VERSION`."""
    try:
        return _distribution_version("scufris")
    except PackageNotFoundError:  # pragma: no cover - packaged always has metadata
        return UNKNOWN_VERSION


#: Resolved once at import; the installed metadata cannot change under a
#: running process.
__version__ = scufris_version()
