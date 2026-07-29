"""Shared shapes for the host inspection reports.

The load-bearing type here is ``Availability``. Every report embeds one, so a
report that could not be produced says WHY on the model itself instead of
raising, returning ``None``, or - worst of the three - returning an empty list
that reads to a model and to an operator as "nothing wrong".

That distinction is the whole contract of this package: an empty result and a
broken one must never render the same. ``Availability.ok`` with no rows means
"asked, answered, nothing there"; ``Availability.ok is False`` means "not
answered, here is what stopped it".
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .run import CommandResult


class Availability(BaseModel):
    """Whether a report holds real data, and the reason when it does not."""

    ok: bool = True
    reason: str = ""
    # Set when the data IS present but only partially - e.g. socket owners that
    # need privilege to resolve. The report is usable; the caveat must be shown.
    caveat: str = ""

    @classmethod
    def unavailable(cls, reason: str) -> Availability:
        return cls(ok=False, reason=reason)

    @classmethod
    def of(cls, result: CommandResult) -> Availability:
        """Derive availability from a command, carrying its reason across."""
        return cls() if result.ok else cls(ok=False, reason=result.reason())

    def line(self) -> str:
        """The single line a text renderer prints for this availability.

        Empty when the report is fully available and uncaveated, so a renderer
        can unconditionally emit it and get nothing in the healthy case.
        """
        if not self.ok:
            return f"unavailable: {self.reason}"
        return f"note: {self.caveat}" if self.caveat else ""


class Report(BaseModel):
    """Base for every host report: it always knows whether it holds data."""

    available: Availability = Field(default_factory=Availability)

    @property
    def ok(self) -> bool:
        return self.available.ok


class Bounded(BaseModel):
    """Mixin state for a report whose rows were capped.

    ``truncated`` exists so a caller can never mistake "the first 200 lines" for
    "all the lines". Renderers must print the marker; a silently cut list is the
    same lie as a silently empty one.
    """

    truncated: bool = False
    limit: int = 0
    total_seen: int = 0

    def truncation_marker(self) -> str:
        if not self.truncated:
            return ""
        seen = f"{self.total_seen}+" if self.total_seen else "more"
        return f"... truncated to {self.limit} of {seen} - narrow the query"


def rejects_option(value: str) -> bool:
    """True when ``value`` would be read as an OPTION rather than an operand.

    ``shell=False`` with an explicit argv stops shell injection, but not option
    injection: a positional argument beginning with ``-`` is still parsed as a
    flag by the command it is handed to. Measured on this host, a unit "pattern"
    of ``-Hsomeone@elsewhere`` makes ``systemctl`` open an outbound SSH
    connection to a caller-chosen host using the service user's credentials.

    Every positional in this package is therefore passed after a ``--``
    separator AND screened with this, because these arguments can come from a
    model that has just read attacker-influenced text (a journal line, a unit
    description). Two independent guards, since the whole package's premise is
    that reading the host cannot do anything.
    """
    return value.startswith("-")


def clamp(value: int, *, default: int, maximum: int, minimum: int = 1) -> int:
    """Force a caller-supplied limit into the configured range.

    A model asking for 100000 journal lines gets the cap, not the lines, and the
    report says it was truncated. Non-positive means "use the default" rather
    than "unbounded" - there is no unbounded path on purpose.
    """
    if value <= 0:
        value = default
    return max(minimum, min(value, maximum))
