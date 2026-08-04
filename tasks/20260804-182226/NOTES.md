# Understanding: the Lane 2 deliverable

## What changes

Lane 2 becomes something that runs. Two artifacts: `examples/operator_decision.py`
and `tasks/20260801-154211/approval.html`.

## Surfaces

- `examples/operator_decision.py` - NEW, unlike Lane 1's, where the member gate
  had already forced the file into the first task. No new package lands in Lane
  2, so nothing forces this one early. It is genuinely this task's to write.
- `tasks/20260801-154211/approval.html`.

## Data and interfaces

No new tables and no new package surface. Consumes what the other four built.

## Sketches

```
  demo output, one screen:

    proposal p1 asked on: web, telegram
      web       card shown
      telegram  card shown

    telegram: operator approves
      -> event seq 12, actor operator
      -> web card resolves     (same event, other channel)
      -> apply runs

    --- replay the same delivery ---
      telegram  key (telegram, c1, 12) exists -> no second card
      web       key (web, c1, 12) exists      -> no second card

    --- refusal ---
      authorize(conn, c1, seq=11)   # seq 11 was said by agent:planner
      PermissionError: ... was said by agent:planner, and only an operator
      event may authorize; what any other party says is a quotation

    --- crash between event and apply ---
      event committed, apply interrupted
      hostd:  proposal p1 still PENDING
      log:    operator approved at seq 12
      -> recoverable
```

Every line is also an assertion.

## Shape

The fourth section is the one worth writing carefully. The reversed write order
is the least visible thing Lane 2 does and the easiest to regress, because
nothing about the happy path looks different. A demo that interrupts between the
committed event and the apply, and then shows both halves of the recoverable
state, is the only artifact that makes the change legible.

## Consequences and open questions

- **Assertions, not decoration.** `tests/test_examples.py` judges by EXIT CODE.
  This is recorded as a lane risk in the epic and it lands here.
- **Open:** how to interrupt between the event and the apply in an offline
  script without contriving it so hard that the demo proves nothing. An
  injected failure in a fake hostd is probably honest enough - the real crash
  and the injected one leave the same two facts behind - but it is worth stating
  in the script that this is simulated.
- **Open:** whether `approval.html` also covers the FLOW gate subject, which
  does not exist until Lane 4. The mechanism is one with two subjects, and
  explaining half of it is close to explaining none of it. Leaning: document
  both, mark the flow half as not-yet-built.
