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
action messages) and to the dashboard as text. `DigestStore` keeps the last few in
the state database so a restart does not lose yesterday's, which is also what makes
"did the 15-minute check fire" answerable when the answer was silence.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import Connection, Row, insert, select
from sqlalchemy import delete as sql_delete
from sqlalchemy import update as sql_update

from .checks import CheckRun, CheckState
from .db import Database
from .db.models import DigestRow

logger = logging.getLogger(__name__)

# How many digests stay readable. Small on purpose: this is a record of the recent
# past, not a metrics store - the audit log and the host page cover the rest.
MAX_DIGESTS = 30


class Digest(BaseModel):
    """One rendered digest, with enough structure for the dashboard to sort it.

    ``id`` is the row key, assigned by :meth:`DigestStore.add` and None until then:
    delivery is recorded against the ROW, and object identity stopped being enough
    to name one when the store became a table.
    """

    id: int | None = None
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
    """The recent digests, in the state database.

    Bounded and append-only in practice: a digest is written once, then updated in
    place only to record whether delivery succeeded. Each method is ONE unit of
    work on the app's one transactional boundary, and every one is SYNCHRONOUS -
    the scheduled-check pass that calls them is ``async def`` and offloads each
    call with ``asyncio.to_thread``.

    The bound is enforced by DELETING the oldest rows inside the insert's own
    transaction, rather than by a bounded deque: the file is the truth now, so a
    bound the process holds in memory would not be a bound at all.
    """

    def __init__(self, db: Database, *, max_digests: int = MAX_DIGESTS) -> None:
        self._db = db
        self._max = max_digests

    def add(self, digest: Digest) -> Digest:
        """Write one digest, assign its id, and drop whatever fell off the end."""
        values = digest.model_dump(exclude={"id"})
        values["states"] = json.dumps(digest.states)
        with self._db.transaction() as conn:
            assigned = conn.execute(
                insert(DigestRow).values(**values).returning(DigestRow.id)
            ).scalar_one()
            self._reap(conn)
        return digest.model_copy(update={"id": assigned})

    def _reap(self, conn: Connection, /) -> None:
        """Drop everything past the newest ``max_digests``, on an OPEN connection.

        Inside the insert's transaction, so the store is never momentarily over
        its bound and a failed insert reaps nothing.
        """
        keep = select(DigestRow.id).order_by(DigestRow.id.desc()).limit(self._max)
        conn.execute(sql_delete(DigestRow).where(DigestRow.id.not_in(keep)))

    def mark_delivered(self, digest: Digest, *, error: str = "") -> Digest:
        """Record the delivery outcome on a digest already in the store.

        Delivery is recorded on the DIGEST rather than only on the schedule, because
        "the machine was fine at 08:00 but you were not told" is a different fact
        from "the 08:00 run failed".

        Keyed on ``digest.id``, which :meth:`add` assigned. Object identity is no
        longer enough to name a row, and a digest that was never added - or that
        has since been reaped - updates nothing rather than resurrecting itself.
        """
        if digest.id is None:
            raise ValueError("this digest has not been added to the store")
        with self._db.transaction() as conn:
            conn.execute(
                sql_update(DigestRow)
                .where(DigestRow.id == digest.id)
                .values(delivered=not error, delivery_error=error)
            )
        return digest.model_copy(
            update={"delivered": not error, "delivery_error": error}
        )

    def latest(self) -> Digest | None:
        with self._db.transaction() as conn:
            row = conn.execute(
                select(DigestRow.__table__).order_by(DigestRow.id.desc()).limit(1)
            ).first()
        return None if row is None else _record(row)

    def last_states(self) -> dict[str, str]:
        """The per-check states of the most recent digest, for the CHANGED section."""
        latest = self.latest()
        return dict(latest.states) if latest is not None else {}

    def list(self) -> list[Digest]:
        """Newest first - a record is read from the top."""
        with self._db.transaction() as conn:
            rows = conn.execute(
                select(DigestRow.__table__).order_by(DigestRow.id.desc())
            ).all()
        return [_record(row) for row in rows]


def _record(row: Row[Any]) -> Digest:
    """The pydantic record for one selected row. Nothing else leaves the store."""
    fields = dict(row._mapping)
    fields["states"] = json.loads(fields["states"])
    return Digest.model_validate(fields)
