# Close the three open round-2 findings on the delivery contract

- PRIORITY: 40
- TAGS: feature,v0.2.0,lane1,chat
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-115319

## Story

As the operator, I want the delivery contract to say what the code does, so
that Lane 2's channel is written against a true docstring and does not carry a
`LookupError` it was told could not happen.

## Steps

- [ ] R2.1 - `packages/chat/src/scufris_chat/store.py:257`. R1.1 and R1.6
      together falsified `confirm_delivery`'s "No correct caller reaches it -
      every one gates its send behind a `True` from `claim_delivery`, which
      hands back only a row it left `claimed`". A re-claim hands `True` back for
      a row it did NOT leave claimed, so two overlapping passes over one channel
      both send and the second confirm raises into the caller loop. Reproduced:
      `pass A claim: True / pass B claim: True / pass A confirm: ok / pass B
      confirm RAISED`. Either correct the docstring and DECISION.md section 6 to
      state the contract that now holds, or make `confirm_delivery` tolerant of
      an already-`confirmed` row and record why. Whichever is chosen needs a
      test for the overlapping pass, which nothing covers today.
- [ ] R2.2 - `packages/chat/src/scufris_chat/store.py:228`. The conflict-loser
      branch is unreachable (engine begins are immediate,
      `packages/core/src/scufris_core/engine.py:268`), untested, and would not
      answer if reached: under a deferred begin the loser takes
      `OperationalError` on the INSERT rather than `rowcount == 0`, and the
      re-SELECT runs on the snapshot that already answered `None`, so
      `scalar_one()` raises `NoResultFound`. The docstring's "the answer is true
      by construction" is not delivered. Either drop lines 223-232 back to
      `return True` and restore the honest precondition, or keep the INSERT and
      claim only what it does - one write instead of a read-then-write.
- [ ] R2.3 - `packages/chat/src/scufris_chat/README.md:213,238`. Two orphan
      part-lines mid-paragraph left by R1.15's reflow: `checks what the` and
      `recorded choice, not an`. Re-wrap both paragraphs whole.

## Definition of Done

- Two overlapping delivery passes over one channel behave the way
  `confirm_delivery`'s docstring and DECISION.md section 6 say they do
  (test: a new case in `packages/chat/tests/test_chat_delivery.py` driving
  claim, claim, confirm, confirm)

  cmd: `pytest packages/chat/tests -q`

- `claim_delivery` has no branch that cannot be reached, or none whose docstring
  claims more than it delivers
  (manual: read `claim_delivery` end to end and confirm every branch has a
  caller path or a test)
- Both README paragraphs wrap whole at 80 columns
  (test: `awk 'length > 80' packages/chat/src/scufris_chat/README.md` lists
  only table rows)

## Notes

- Source: `tasks/20260804-115319/REVIEW.md` round 2. All three were APPROVEd
  open as MINOR/NIT; none blocks the delivery table.
