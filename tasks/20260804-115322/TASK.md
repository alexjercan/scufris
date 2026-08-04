# Prove Lane 1 with the conversation demo and the chat explainer

- PRIORITY: 96
- TAGS: feature, v0.2.0, lane1, chat, deliverable
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-115256, 20260804-115319, 20260804-115320, 20260804-115321

## Story

As the maintainer, I want Lane 1 to end in something I can run and read, so
that "the conversation exists" is a claim I have watched succeed rather than a
row of green test names.

Note what this task is NOT. `examples/chat_conversation.py` already exists from
the first Lane 1 task - the member gate in `tests/test_examples.py` requires it
the moment `packages/chat` appears. This task does not create the example. It
makes the example prove the WHOLE LANE, and writes the explainer.

## Steps

- [ ] Grow `examples/chat_conversation.py` into the lane demo: one conversation
      printed as a rich transcript, a colour per typed actor, `event_seq` shown,
      causation rendered as a tree.
- [ ] Mid-script, switch backend and re-print. The semantic transcript is
      identical; the provider session id is not. Both facts are visible on
      screen AND asserted.
- [ ] Put at least one assertion behind every claim the output makes. The
      example gate judges by EXIT CODE, so a rich table nobody asserts on is
      decoration that still exits 0. The rendering is for the operator; the
      assertions are for the gate.
- [ ] Write `tasks/20260801-154211/chat.html` beside `architecture.html`: the
      event model, the four owned records and their owners, the settled per-turn
      granularity, and the retention non-decision stated as a choice.
- [ ] Confirm the boundary and example gates are green for the new member.

## Definition of Done

- The demo runs offline in a clean checkout and its assertions carry its claims
  (cmd: `python -m pytest tests/test_examples.py`).
- `packages/chat` imports only `core`
  (cmd: `python -m pytest tests/test_package_boundaries.py`).
- `chat.html` states the event model, the owners, the granularity decision and
  the retention non-decision
  (manual: user reads chat.html and agrees it explains the lane).
- The demo is legible to someone who has not read the code
  (manual: user runs the demo and follows what happened from its output alone).

## Notes

- Lane 1 deliverable of `tasks/20260801-154211/TASK.md`. The lane is not done
  until this record is.
- Depends on all four Lane 1 build tasks.
- Deliberately a separate record: folded into the last build task, this is the
  part that gets dropped under schedule pressure and nobody sees it happen.
