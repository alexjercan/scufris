# Decouple the host approval decision from the privileged host client

- PRIORITY: 94
- TAGS: feature, v0.2.0, lane2, hostctl
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-182222, 20260804-141639

## Story

As the operator, I want a host action's approval to be a conversation event that
commits before the action runs, so that a crash between deciding and applying
loses the apply - which is recoverable - rather than the record that I approved
it, which is not.

## Steps

- [ ] Move the decision half out of `HostApprovalService`
      (`packages/hostctl/src/scufris_hostctl/approvals.py`, 549 lines) to the
      shared operator decision mechanism. Leave propose, apply, deny and audit
      behind: `hostctl` exists to talk to `hostd`.
- [ ] Replace `actor: str` on `approve()` with a minted `OperatorDecision`. The
      current signature takes a string any caller can fabricate; the token
      cannot be constructed outside `authorize`.
- [ ] Keep `approve()` the only caller of `apply`. The module docstring's
      guarantee - "an action with no approval has no route to execution, not
      because a check refuses it but because nothing else calls it" - is
      preserved and strengthened, not traded away.
- [ ] REVERSE the write order: event first, then apply. Today `_fire` runs after
      the row commits (`approvals.py:415`), so a crash between them loses the
      conversation event permanently. Event-first loses only the apply, which is
      recoverable - the log says an operator approved it and `hostd` still holds
      the proposal pending.
- [ ] Give the announcement a real idempotency key from `chat`'s `delivery`
      table in place of `TelegramApprovals._announced`
      (`scufris/telegram/approvals.py:79`), an in-memory `OrderedDict` capped at
      `MAX_TRACKED_ACTIONS` that dies on restart.
- [ ] Record the answer to the epic's open question in `DECISION.md`: are host
      approvals conversation events? The DECISION is; the proposal is not.

## Definition of Done

- An approval writes its conversation event and applies in an order where a
  crash between them is recoverable
  (test: `test_crash_after_event_leaves_the_proposal_pending`).
- `approve()` cannot be called without a minted decision
  (test: `test_approve_requires_an_operator_decision`).
- Nothing but `approve` calls `apply`
  (test: `test_apply_has_one_caller`).
- A redelivered announcement does not produce a second card, across a restart
  (test: `test_announcement_is_idempotent_across_restart`).
- The existing approval flow example still runs
  (cmd: `python -m pytest tests/test_examples.py`).

## Notes

- Depends on the `OperatorDecision` home task and on `chat`'s `delivery` table.
- `tasks/20260804-141639` falsified part of `confirm_delivery`'s docstring;
  land that first so this is written against a true contract.
- This re-opens a package the carve epic called complete. That risk is recorded
  in `tasks/20260801-154211/TASK.md`.
- Lane 2 of `tasks/20260801-154211/TASK.md`.
