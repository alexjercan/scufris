"""The app's side of a host action: who asked, who decided, and what happened.

The privileged half of the record is the helper's (root-written, append-only,
independent of anything the app can rewrite). This is the OTHER half - the
request side the spike left as app state: what has been proposed, what the
operator decided, and which run is carrying it out, so the dashboard can show a
queue and a chat client can be told "waiting for you".

It is a DECISION JOURNAL in the state database, and bounded. The PROPOSAL half is
still short-lived and still the helper's - ``refresh_pending`` reconciles the
pending set from the socket, and nothing here deletes a record the helper stopped
listing. What durability buys is the other half: "what did I approve, and why did
I deny that" outlives the minutes the helper keeps a proposal, and a restart mid
queue no longer answers "there was never any such action".
"""

from __future__ import annotations

import json
import time
from enum import StrEnum
from typing import Any, Callable

from pydantic import BaseModel, computed_field
from sqlalchemy import Connection, Row, func, insert, select
from sqlalchemy import delete as sql_delete
from sqlalchemy import update as sql_update

from scufris_hostd import ProposalView, ResultFrame, RiskClass

from .db import Database
from .db.models import HostActionRow

# Bounded like any per-request registry: the oldest decided actions are dropped
# first, and a pending one is never dropped ahead of a decided one.
MAX_ACTIONS = 200


class Decision(StrEnum):
    """What the operator has said about a proposal, if anything yet."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


class ConfirmationStyle(StrEnum):
    """How much the operator must do to say yes.

    ONE_WAY is the strong path: the approve call must carry an explicit
    acknowledgement token, so no surface can render the action as an ordinary
    one-tap approve. ORDINARY is everything else - which still SHOWS whether the
    action can be undone (``Confirmation.no_undo`` and ``undo``); the styles differ
    in required friction, not in honesty.
    """

    ORDINARY = "ordinary"
    ONE_WAY = "one_way"


class Confirmation(BaseModel):
    """What confirming ONE action requires, computed once for both surfaces.

    A surface renders this; it does not decide it. That is what stops the two from
    drifting into two different ideas of "are you sure", and it is why the
    acknowledgement is a token the SERVICE checks rather than a checkbox a client
    can decide to skip.
    """

    style: ConfirmationStyle
    risk: RiskClass
    # A phrase for the risk class, so a service restart and a system switch do not
    # read identically in a queue.
    risk_label: str
    # How it can be undone, or the statement that it cannot. Never empty, and
    # never softened - it is the sentence the operator is deciding against.
    undo: str
    # Whether an undo exists AT ALL, straight from the helper. Separate from the
    # style on purpose: a restart of a running unit has no undo (the process it was
    # running is gone) but is not a destructive act, so a surface must be able to
    # SAY "no undo" without the strong path being triggered by it.
    no_undo: bool = False
    # The token an approve MUST carry for a one-way action; empty for an ordinary
    # one. The value is the action's own kind, so the operator confirms by naming
    # the thing they are doing.
    acknowledge: str = ""

    @property
    def one_way(self) -> bool:
        return self.style is ConfirmationStyle.ONE_WAY


# A phrase for each risk CLASS. Deliberately about the class and not about the
# individual action: whether THIS action can be undone is `Confirmation.undo`,
# straight from the helper, and a label claiming "reversible" next to an undo line
# saying "this cannot be undone" is the kind of contradiction an operator stops
# reading. (Measured in `examples/host_agent.py`: restarting a running unit is R1
# and has no undo.)
_RISK_LABEL: dict[RiskClass, str] = {
    RiskClass.R1: "service control - changes a unit's runtime state, destroys no data",
    RiskClass.R2: "disposable cleanup - ONE-WAY, what it frees does not come back",
    RiskClass.R3: (
        "system configuration - switches the whole system, back to a recorded "
        "generation if needed"
    ),
}


def confirmation_for(proposal: ProposalView) -> Confirmation:
    """What approving ``proposal`` requires.

    The strong path is for an action that DESTROYS something irrecoverably:
    ``reversal.possible`` is false AND the action is not mere service control.

    The first version of this rule used ``reversal.possible`` alone, on the
    reasoning that a future irreversible verb would then inherit the strong path
    without anyone remembering to list it. Running it against the real helper
    refuted that: for R1, "no undo" is the NORMAL answer, not the alarming one -
    restarting or reloading an active unit ends where it started, so there is no
    state to restore (``hostd/preview.py``), and starting an already-active unit
    reports no undo because it changed nothing. Keying on reversibility alone made
    every service restart demand a typed acknowledgement, which is both wrong about
    the risk and self-defeating: a warning that fires on the routine act is the
    reason nobody reads the one on ``gc_store``.

    So R1 - service control, which changes runtime state and destroys no data - is
    carved out and confirmed ordinarily, with its no-undo sentence shown as written.
    Everything else that declares itself irreversible takes the strong path: the
    disposable cleanups (R2) whose freed store paths do not come back, and any
    other class where irreversibility is an anomaly rather than the norm (an R3
    whose previous generation could not be recorded would be exactly that).
    """
    reversal = proposal.reversal
    destructive = not reversal.possible and proposal.risk is not RiskClass.R1
    return Confirmation(
        style=ConfirmationStyle.ONE_WAY if destructive else ConfirmationStyle.ORDINARY,
        risk=proposal.risk,
        risk_label=_RISK_LABEL.get(proposal.risk, str(proposal.risk)),
        undo=reversal.summary,
        no_undo=not reversal.possible,
        acknowledge=str(proposal.kind) if destructive else "",
    )


class HostActionRecord(BaseModel):
    """One proposed action as the dashboard and the API see it."""

    proposal: ProposalView
    decision: Decision = Decision.PENDING
    decided_by: str = ""
    decided_at: float | None = None
    reason: str = ""
    # The supervisor run carrying out an approved action, so a client can attach
    # to its output stream.
    run_id: str | None = None
    result: ResultFrame | None = None
    error: str = ""

    @property
    def id(self) -> str:
        return self.proposal.id

    @computed_field  # type: ignore[prop-decorator]
    @property
    def confirmation(self) -> Confirmation:
        """What approving this action requires - its risk class in words, the undo
        sentence, and the acknowledgement token a one-way action needs.

        Computed and carried INLINE on every record rather than fetched per row: a
        queue render needs the risk of each row to render it at all, and both
        approval surfaces must render the same requirement rather than each deciding
        how much friction an action deserves.
        """
        return confirmation_for(self.proposal)


class UnknownAction(KeyError):
    """No such action id in this app's registry."""


class AlreadyDecided(RuntimeError):
    """The operator has already answered this proposal.

    A second decision is refused rather than applied: an approval is a single
    act, and "approve twice" must not become "run twice" at any layer.
    """


class HostActionStore:
    """The app's bounded decision journal of proposed host actions.

    Every method is ONE unit of work on the app's one transactional boundary, and
    every one is SYNCHRONOUS: :class:`~scufris.host_approvals.HostApprovalService`
    is async throughout, so it offloads each call with ``asyncio.to_thread``.

    :meth:`_decide` is the reason the boundary matters here. It reads the record,
    checks that it is still pending, and writes the decision INSIDE one immediate
    transaction, so two surfaces answering the same proposal at the same moment
    produce one decision and one :class:`AlreadyDecided`. The read-check-mutate
    this replaced could not: both readers saw PENDING, and both approved.
    """

    def __init__(
        self,
        db: Database,
        *,
        max_actions: int = MAX_ACTIONS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._db = db
        self._max = max_actions
        self._now = clock

    def put(self, proposal: ProposalView) -> HostActionRecord:
        """Record a proposal, or re-record one whose id is already known.

        Upsert rather than insert: the helper reissues a proposal id across a
        `refresh_pending` on a restart, and a record already decided keeps its
        decision - the proposal snapshot is the only part the helper owns.
        """
        record = HostActionRecord(proposal=proposal)
        with self._db.transaction() as conn:
            existing = _row(conn, proposal.id)
            if existing is not None:
                conn.execute(
                    sql_update(HostActionRow)
                    .where(HostActionRow.id == proposal.id)
                    .values(proposal=proposal.model_dump_json())
                )
                return _record(existing).model_copy(update={"proposal": proposal})
            # `seq` is assigned here, inside the inserting transaction: the begin
            # is immediate, so two proposals arriving together cannot claim one
            # number. See `HostActionRow` for why it is not the rowid.
            nxt = conn.execute(select(func.coalesce(func.max(HostActionRow.seq), 0)))
            conn.execute(
                insert(HostActionRow).values(
                    seq=nxt.scalar_one() + 1, **_values(record)
                )
            )
            self._reap(conn)
        return record

    def get(self, action_id: str) -> HostActionRecord:
        with self._db.transaction() as conn:
            return _record(_require(conn, action_id))

    def list(self) -> list[HostActionRecord]:
        """Newest first - a queue is read from the top."""
        with self._db.transaction() as conn:
            rows = conn.execute(
                select(HostActionRow.__table__).order_by(HostActionRow.seq.desc())
            ).all()
        return [_record(row) for row in rows]

    def approve(self, action_id: str, *, operator: str) -> HostActionRecord:
        return self._decide(action_id, Decision.APPROVED, operator=operator)

    def deny(
        self, action_id: str, *, operator: str, reason: str = ""
    ) -> HostActionRecord:
        # The reason goes in with the decision rather than as a second mutation:
        # a denial whose reason landed separately can be read, and reported to the
        # requesting agent, without it.
        return self._decide(
            action_id, Decision.DENIED, operator=operator, reason=reason
        )

    def _decide(
        self, action_id: str, decision: Decision, *, operator: str, reason: str = ""
    ) -> HostActionRecord:
        """Read, check and write ONE decision inside ONE transaction."""
        decided_at = self._now()
        with self._db.transaction() as conn:
            record = _record(_require(conn, action_id))
            if record.decision is not Decision.PENDING:
                raise AlreadyDecided(
                    f"this action was already {record.decision} by "
                    f"{record.decided_by or 'someone'}"
                )
            conn.execute(
                sql_update(HostActionRow)
                .where(HostActionRow.id == action_id)
                .values(
                    decision=str(decision),
                    decided_by=operator,
                    decided_at=decided_at,
                    reason=reason,
                )
            )
        return record.model_copy(
            update={
                "decision": decision,
                "decided_by": operator,
                "decided_at": decided_at,
                "reason": reason,
            }
        )

    def attach_run(self, action_id: str, run_id: str) -> HostActionRecord:
        return self._amend(action_id, {"run_id": run_id})

    def finish(
        self,
        action_id: str,
        *,
        result: ResultFrame | None = None,
        error: str = "",
    ) -> HostActionRecord:
        values: dict[str, Any] = {}
        if result is not None:
            values["result"] = result.model_dump_json()
        if error:
            values["error"] = error
        return self._amend(action_id, values)

    def refresh(self, action_id: str, proposal: ProposalView) -> HostActionRecord:
        """Replace the held proposal snapshot (the helper owns its state)."""
        return self._amend(action_id, {"proposal": proposal.model_dump_json()})

    def _amend(self, action_id: str, values: dict[str, Any]) -> HostActionRecord:
        """Apply ``values`` to one existing record and return what it became.

        Read and write in one transaction so the record returned is the record
        written, rather than a snapshot another writer may already have moved on.
        """
        with self._db.transaction() as conn:
            _require(conn, action_id)
            if values:
                conn.execute(
                    sql_update(HostActionRow)
                    .where(HostActionRow.id == action_id)
                    .values(**values)
                )
            return _record(_require(conn, action_id))

    def _reap(self, conn: Connection, /) -> None:
        """Hold the bound, on an OPEN connection, decided actions first.

        A pending action is never dropped ahead of a decided one: it is the one
        the operator has still to answer. When everything is pending the oldest
        goes anyway rather than growing without bound - the helper still holds it,
        so nothing is lost that a refresh or an audit read cannot recover.
        """
        over = conn.execute(
            select(func.count()).select_from(HostActionRow)
        ).scalar_one()
        over -= self._max
        if over <= 0:
            return
        doomed = (
            select(HostActionRow.id)
            .order_by(
                # Decided first, then oldest - `decision` sorts PENDING last only by
                # accident of spelling, so the ordering is made explicit.
                (HostActionRow.decision == str(Decision.PENDING)).asc(),
                HostActionRow.seq.asc(),
            )
            .limit(over)
        )
        conn.execute(sql_delete(HostActionRow).where(HostActionRow.id.in_(doomed)))


def _row(conn: Connection, action_id: str) -> Row[Any] | None:
    return conn.execute(
        select(HostActionRow.__table__).where(HostActionRow.id == action_id)
    ).first()


def _require(conn: Connection, action_id: str) -> Row[Any]:
    row = _row(conn, action_id)
    if row is None:
        raise UnknownAction(action_id)
    return row


def _values(record: HostActionRecord) -> dict[str, Any]:
    """One record as columns. ``proposal`` and ``result`` stay JSON text - they are
    the HELPER's protocol types, and shredding them would make every helper change
    a database migration."""
    return {
        "id": record.id,
        "proposal": record.proposal.model_dump_json(),
        "decision": str(record.decision),
        "decided_by": record.decided_by,
        "decided_at": record.decided_at,
        "reason": record.reason,
        "run_id": record.run_id,
        "result": None if record.result is None else record.result.model_dump_json(),
        "error": record.error,
    }


def _record(row: Row[Any], /) -> HostActionRecord:
    """The pydantic record for one selected row. Nothing else leaves the store."""
    fields = dict(row._mapping)
    fields.pop("seq", None)
    fields.pop("id", None)  # derived from the proposal; not a settable field
    fields["proposal"] = json.loads(fields["proposal"])
    fields["result"] = (
        None if fields["result"] is None else json.loads(fields["result"])
    )
    return HostActionRecord.model_validate(fields)


def render_action(record: HostActionRecord) -> str:
    """One action as plain text: what it does, what it would change, how to undo.

    Used by the propose endpoint's text rendering (so an agent shows the operator
    the real preview rather than its own paraphrase) and by
    ``examples/host_action.py``. The dashboard has its own surface; this is the
    text version that has to be right first.

    Two things are always present and never softened: the LABEL saying what kind
    of preview this is, and the reversal line saying how it can be undone or
    that it cannot.
    """
    proposal = record.proposal
    lines = [
        f"host action {proposal.id}",
        f"  what:     {proposal.summary}",
        f"  risk:     {proposal.risk} ({proposal.kind})",
    ]
    # Every command, in order. An action with two steps has two lines here, so
    # "what am I approving" is never a summary of a sequence.
    for index, step in enumerate(proposal.steps, start=1):
        prefix = f"  command {index}:" if len(proposal.steps) > 1 else "  command: "
        lines.append(f"{prefix} {' '.join(step.argv)}")
        if step.label and len(proposal.steps) > 1:
            lines.append(f"             ({step.label})")
    lines += [
        f"  decision: {record.decision}"
        + (f" by {record.decided_by}" if record.decided_by else ""),
        "",
        f"PREVIEW ({proposal.preview.kind}): {proposal.preview.label}",
    ]
    availability = proposal.preview.available.line()
    if availability:
        lines.append(f"  {availability}")
    lines.extend(f"  {line}" for line in proposal.preview.lines)
    lines.append("")
    if proposal.reversal.possible:
        lines.append(f"UNDO: {proposal.reversal.summary}")
    else:
        lines.append(f"NO UNDO: {proposal.reversal.summary}")
    if record.result is not None:
        outcome = "succeeded" if record.result.ok else "FAILED"
        progress = (
            f", {record.result.steps_completed}/{record.result.steps_total} steps"
            if record.result.steps_total > 1
            else ""
        )
        lines.append(
            f"RESULT: {outcome} ({record.result.outcome}, exit "
            f"{record.result.returncode}{progress})"
        )
        if record.result.detail:
            lines.append(f"  {record.result.detail}")
    if record.error:
        lines.append(f"ERROR: {record.error}")
    return "\n".join(lines)
