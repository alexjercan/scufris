# Decouple the host approval decision from the privileged host client

- PRIORITY: 94
- TAGS: feature, v0.2.0, lane2, hostctl
- ACTIVITY: UNDERSTANDING
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
- [ ] REVERSE the write order: the conversation event commits BEFORE the apply
      starts. Corrected 2026-08-04 after review - the order today is claim the
      decision (`approvals.py:396`), START the apply (`:407`), attach the run
      (`:414`), then notify (`:415`). The apply is already running before the
      hook, so "event before `_fire`" was the wrong target: the event has to
      land before `:396`, not before `:415`. The defect is unchanged - a crash
      after the claim commits loses the conversation event permanently - but the
      fix is earlier in the method than this record first said.
- [ ] Make the decision row and the conversation event ONE unit of work, which
      `tasks/20260729-220835/DECISION.md` section 4 requires and which is
      currently impossible: `chat` takes the caller's open `Connection`,
      `HostActionStore` opens its own (`actions.py:222,244,249,273,326`), and
      `Database.transaction()` refuses to nest
      (`packages/core/src/scufris_core/engine.py:186`). So `HostActionStore`
      must learn to accept a `Connection`. That is a signature change through a
      module the carve called complete and it is the real cost of this task.
- [ ] Give the announcement a durable idempotency key from `chat`'s `delivery`
      table. Corrected after review: this does NOT simply replace
      `ApprovalSurface._announced` (`scufris/telegram/approvals.py:79`; the
      class is `ApprovalSurface`, not `TelegramApprovals` - the name in
      `DECISION.md:87` and in this record's first draft is a fiction).
      `_announced` maps `action_id -> [(chat_id, message_id)]` and its consumer
      is `announce_decision` (`:130`), which EDITS the existing card. `delivery`
      has no message-reference column (`models.py:238-243`), so it can dedupe
      the send and cannot resolve the card. Decide which: `delivery` gains a
      channel-local handle, or the message map stays per-channel and only the
      send is deduped.
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
