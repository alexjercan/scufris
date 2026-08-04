# Decide where OperatorDecision lives before a second package consumes it

- PRIORITY: 95
- TAGS: feature, v0.2.0, lane2, chat
- ACTIVITY: UNDERSTANDING
- GATES: -
- RESOLUTION: -
- PARENT: 20260801-154211

## Story

As the maintainer, I want one decided home for `OperatorDecision` before a
second package consumes it, so that the host approval decoupling and the flow
guard are written against a boundary that was chosen rather than one that
happened.

`packages/chat/src/scufris_chat/README.md` section 3.1 books this move
explicitly: "`OperatorDecision` lives here rather than in `scufris_core` until a
second package consumes it... The move is booked, not forgotten." This lane is
where the second consumer arrives.

## Steps

- [x] Choose the home. DECIDED 2026-08-04 by the maintainer: **option B, the
      type stays in `chat` and `hostctl` gains a declared edge to it.** Chosen
      to keep `core` lean, which is a standing design preference. The four
      options, since the first draft of this record named a count and no
      options:
      **(A) type in `core`, mint in `chat`** - the epic's assumption. DEAD, and
      both independent reviewers killed it with the same argument: `authorize`
      must stay in `chat`, so `chat` must construct `core`'s type, which needs
      the module-private `_WITNESS`; the facade rule forbids reaching a
      sibling's private module, so the sentinel would have to be exported from
      `core`'s public surface - at which point anyone can mint and the property
      `test_chat_authority.py` asserts is gone. A trades the lane's security
      guarantee for a graph edge.
      **(B) type stays in `chat`, declare `hostctl -> chat`** - CHOSEN.
      **(C) `approve()` leaves `hostctl` entirely**, decision-taking caller
      above it. Rejected: it is the only option where `hostctl` shrinks, but
      "nothing but `approve` calls `apply`" would have to be re-established at a
      new home, and that property is worth more than the shrink.
      **(D) abstract capability in `core`, concrete in `chat`, root passes the
      open `Connection`** - raised by review, not in the first draft. Rejected
      for the same reason the epic rejected Protocol ports: it is more machinery
      than B for a boundary with two consumers.
- [ ] Record all four and the rejection reasons in `DECISION.md`. B's cost is
      real and gets written down: the privileged host client depends on the
      conversation package.
- [ ] Settle HOW DEEP the edge goes, which is the part B does not answer on its
      own. If `hostctl` imports only the TYPE and the composition root writes
      the event, the edge is thin. If `hostctl` calls `append_event` and
      `claim_delivery` itself, it depends on the chat STORE at runtime, which is
      a much larger edge than "where the type lives". Decide this here, not
      inside the decoupling task.
- [ ] Apply the choice: the module move, the `DECLARED_GRAPH` edit, and the
      `CORE_MODULES` allowlist entry if the type lands in `core`.
- [ ] Keep `authorize` in `chat` whatever is chosen. It reads the `event` table;
      it cannot move.
- [ ] Confirm the boundary tests express the choice rather than tolerate it.

## Definition of Done

- The declared graph matches the real imports after the change
  (cmd: `python -m pytest tests/test_package_boundaries.py`).
- Minting outside `authorize` is still refused, and the witness still cannot be
  re-targeted by `dataclasses.replace`
  (cmd: `python -m pytest packages/chat/tests/test_chat_authority.py`).
- `DECISION.md` names the chosen home, the two rejected options, and what would
  reopen the choice
  (manual: user reads DECISION.md and agrees the rejected options are stated
  fairly).

## Notes

- Blocks the host approval decoupling and Lane 4's flow guard. Both take an
  `OperatorDecision` as an argument, so neither can be written until its import
  path is settled.
- Lane 2 of `tasks/20260801-154211/TASK.md`.
