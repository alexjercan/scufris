# Decide where OperatorDecision lives before a second package consumes it

- PRIORITY: 95
- TAGS: feature, v0.2.0, lane2, chat
- KIND: TASK
- ACTIVITY: PLANNING
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

- [ ] Choose between the three options and record the rejected two in
      `DECISION.md`. They are not equivalent and the epic's assumption - that
      the type lives in `core` - was made before `core` was proven domain-free.
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
