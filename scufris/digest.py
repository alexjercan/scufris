"""The digest: what the checks found, in one message worth reading.

Two rules, both about attention rather than information:

- **The boring case is one line.** Not a table of green ticks: a digest that costs
  nothing to read is a digest that still gets read in week three.
- **It leads with what needs attention, and only speaks when something CHANGED.**
  `watch` renders a message when a check ENTERS or worsens into an attention state,
  or recovers - never for a condition the operator was already told about. A
  96%-full disk stays 96% full for days, and re-sending that every fifteen minutes
  is how a useful feature gets muted (review round 1, R1.1: measured at 96 messages
  a day for a disk that had not moved). Persistent state is what the daily line is
  for, which is why `daily` renders unconditionally.

The rendered text is plain: it goes to Telegram without a parse mode (like the host
action messages) and to the dashboard as text. `DigestStore` keeps the last few on
disk so a restart does not lose yesterday's, which is also what makes "did the
15-minute check fire" answerable when the answer was silence.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .checks import CheckRun, CheckState

logger = logging.getLogger(__name__)

# How many digests stay readable. Small on purpose: this is a record of the recent
# past, not a metrics store - the audit log and the host page cover the rest.
MAX_DIGESTS = 30


class Digest(BaseModel):
    """One rendered digest, with enough structure for the dashboard to sort it."""

    at: float
    schedule: str
    # ok | attention: whether anything wanted the operator when this was written.
    verdict: str
    text: str
    # Whether it was actually delivered, and why not when it was not.
    delivered: bool = False
    delivery_error: str = ""
    # The per-check states, so the dashboard can show the shape without re-reading
    # and the next digest can say what changed.
    states: dict[str, str] = Field(default_factory=dict)


def _stamp(at: float) -> str:
    return datetime.fromtimestamp(at, timezone.utc).astimezone().strftime("%H:%M")


def render_digest(
    run: CheckRun,
    *,
    previous: dict[str, str] | None = None,
    schedule: str = "watch",
    always: bool = False,
) -> Digest | None:
    """Render one digest, or None when there is nothing worth sending.

    ``previous`` is the last digest's per-check states, which is what makes the
    CHANGED section possible. ``always`` is the daily heartbeat: it renders the
    one-line all-clear instead of returning None, so silence is never ambiguous.
    """
    before = previous or {}
    attention = run.attention
    recovered = [
        result
        for result in run.results
        if result.state is CheckState.OK
        and before.get(result.name) not in (None, CheckState.OK.value)
    ]
    states = {result.name: result.state.value for result in run.results}
    # What is NEW: a check whose state differs from the last digest's. A first-ever
    # digest has nothing to compare against, so everything counts as new (`before` is
    # empty) - which is right: the operator has not been told any of it.
    changed = [
        result for result in attention if before.get(result.name) != result.state.value
    ]

    if not always and not changed and not recovered:
        # Nothing has changed since the operator was last told. Silence is the honest
        # answer, and the daily line is where standing conditions get repeated.
        return None

    lines: list[str] = []
    if attention:
        # The lead is the worst thing, in its own words - NAMED, because a headline
        # alone ("no sensors") does not say which check produced it, and a digest
        # whose first line is ambiguous has already lost the reader.
        worst = attention[0]
        lines.append(f"{_stamp(run.at)} - {worst.name}: {worst.headline}")
        # The worst one's detail goes DIRECTLY under its own line - printing it after
        # the other headlines made it read as the detail of whichever check happened
        # to be listed last (seen in `examples/host_digest.py`: the disk's three
        # filesystem lines appeared under the store's headline).
        for line in worst.detail[:4]:
            lines.append(f"    {line}")
        # The rest: one line each. A digest is a pointer, not a report.
        for result in attention[1:]:
            lines.append(f"  - {result.name}: {result.headline}")
    elif recovered:
        lines.append(f"{_stamp(run.at)} - recovered, nothing else needs you")
    else:
        lines.append(f"{_stamp(run.at)} - all clear on {_all_clear(run)}")

    if recovered:
        names = ", ".join(result.name for result in recovered)
        lines.append(f"  recovered since the last digest: {names}")

    if changed and before:
        names = ", ".join(f"{r.name} ({r.state.value})" for r in changed)
        lines.append(f"  new since the last digest: {names}")
    elif attention and before and not always:
        # Reachable only on the daily line's siblings: a `watch` message always has
        # something changed. Kept explicit so a future edit cannot turn silence into
        # a repeat without noticing.
        lines.append("  (unchanged since the last digest)")

    if always and attention:
        # On the daily line, name the checks that are fine too - it is the one
        # message where "everything else is fine" is worth the words.
        fine = [r.name for r in run.results if r.state is CheckState.OK]
        if fine:
            lines.append(f"  fine: {', '.join(fine)}")

    return Digest(
        at=run.at,
        schedule=schedule,
        verdict="attention" if attention else "ok",
        text="\n".join(lines),
        states=states,
    )


def _all_clear(run: CheckRun) -> str:
    """The all-clear's subject: how many checks passed, and any that could not run."""
    ok = sum(1 for result in run.results if result.state is CheckState.OK)
    unavailable = [
        result.name for result in run.results if result.state is CheckState.UNAVAILABLE
    ]
    text = f"{ok} check(s)"
    if unavailable:
        # Never let an unreadable check hide inside an all-clear: it is not a pass.
        text += f" ({', '.join(unavailable)} could not be read)"
    return text


class DigestStore:
    """The recent digests, persisted under the state dir.

    Bounded and append-only in practice: a digest is written once, then updated in
    place only to record whether delivery succeeded. Tolerant load, atomic write -
    the same shape as the other stores here, because a corrupt file must not stop the
    app from booting.
    """

    def __init__(self, state_dir: Path, *, max_digests: int = MAX_DIGESTS) -> None:
        self._path = Path(state_dir) / "digests.json"
        self._max = max_digests
        self._digests: deque[Digest] = deque(maxlen=max_digests)
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("digest store: cannot read %s: %s", self._path, exc)
            return
        rows = raw.get("digests") if isinstance(raw, dict) else None
        if not isinstance(rows, list):
            return
        for row in rows:
            try:
                self._digests.append(Digest.model_validate(row))
            except ValueError as exc:
                logger.warning("digest store: dropping an invalid digest: %s", exc)

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"digests": [digest.model_dump() for digest in self._digests]}
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self._path)

    def add(self, digest: Digest) -> Digest:
        self._digests.append(digest)
        self._persist()
        return digest

    def mark_delivered(self, digest: Digest, *, error: str = "") -> Digest:
        """Record the delivery outcome on a digest already in the store.

        Delivery is recorded on the DIGEST rather than only on the schedule, because
        "the machine was fine at 08:00 but you were not told" is a different fact
        from "the 08:00 run failed".
        """
        digest.delivered = not error
        digest.delivery_error = error
        self._persist()
        return digest

    def latest(self) -> Digest | None:
        return self._digests[-1] if self._digests else None

    def last_states(self) -> dict[str, str]:
        """The per-check states of the most recent digest, for the CHANGED section."""
        latest = self.latest()
        return dict(latest.states) if latest is not None else {}

    def list(self) -> list[Digest]:
        """Newest first - a record is read from the top."""
        return list(reversed(self._digests))
