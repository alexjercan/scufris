# Prove Lane 2 with the operator decision demo and the approval explainer

- PRIORITY: 91
- TAGS: feature, v0.2.0, lane2, deliverable
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-182222, 20260804-182223, 20260804-182224, 20260804-182225

## Story

As the maintainer, I want Lane 2 to end in something I can run and read, so that
"a human can say yes" is a claim I have watched succeed - including the part
where saying yes twice changes nothing.

## Steps

- [ ] Write `examples/operator_decision.py`, offline: a host proposal is asked
      on two channels; answered on one; the other channel's card resolves.
- [ ] REPLAY the same delivery in the same script and show that nothing happens
      twice, with an assertion behind it rather than only in the output.
- [ ] Attempt to mint a decision from an `agent:` event and print the refusal,
      naming the actor.
- [ ] Show the reversed write order concretely: interrupt between the committed
      event and the apply, then show the proposal still pending and the log
      still saying an operator approved it.
- [ ] Write `tasks/20260801-154211/approval.html`: the flow, both subjects - a
      flow gate and a host proposal are one mechanism - and why the write order
      reversed. Diagrams over prose.

## Definition of Done

- The demo runs offline in a clean checkout and its assertions carry its claims
  (cmd: `python -m pytest tests/test_examples.py`).
- The refusal path and the replay path are both exercised, not only described
  (test: `test_operator_decision_example_covers_refusal_and_replay`).
- `approval.html` explains the mechanism, both subjects and the write order
  (manual: user reads approval.html and agrees it explains the lane).
- The demo is legible to someone who has not read the code
  (manual: user runs the demo and follows what happened from its output alone).

## Notes

- Depends on every other Lane 2 task.
- Lane 2 deliverable of `tasks/20260801-154211/TASK.md`. The lane is not done
  until this record is; that is why it is a separate record rather than a step
  inside the last build task.
- One of exactly three HTML explainers scheduled for the release.
