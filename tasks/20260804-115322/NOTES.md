# Understanding: the Lane 1 deliverable

## What changes

Lane 1 stops being a row of green test names and becomes something that runs and
is legible. Two artifacts: the demo and the explainer.

## Surfaces

- `examples/chat_conversation.py` - GROWN, not created. It has existed since
  `20260804-115256` because `EXAMPLES_BY_MEMBER` forces it the moment
  `packages/chat` appears on disk.
- `tasks/20260801-154211/chat.html` - new, beside `architecture.html`.

## Data and interfaces

No new tables, no new package surface. This task consumes what the other four
built.

## Sketches

```
  demo output, one screen:

    conversation c1
    seq  actor        content
      1  operator     "plan the release"          <- colour per actor kind
      2  agent:plan   > "found 18 open bullets"   <- quoted, attributed
      3  system       backend switched: claude -> codex
      4  operator     "go"

    causation:  1 --> 2 --> 4

    --- switching backend ---
    provider session:  sess_abc  ->  rollout_xyz     CHANGED
    semantic transcript:                             IDENTICAL
```

Every line above is also an assertion.

## Shape

**What this task is not.** It does not write the example. It makes the example
prove the WHOLE LANE, and writes the explainer.

**Why it is a separate record.** Folded into the last build task, this is the
part that gets dropped under schedule pressure and nobody sees it happen. As its
own record it shows as open in `tatr frontier 20260801-154211`, so the lane
cannot look finished while it is unproven.

**Assertions behind every claim.** `tests/test_examples.py` judges each example
by its EXIT CODE. A rich table nobody asserts on is decoration that still exits
0 - it would pass the gate forever while the thing it claims to show is broken.
The rendering is for the operator; the assertions are for the gate. This is
recorded as a lane risk in the epic and it lands here.

## Consequences and open questions

- Depends on all four Lane 1 build tasks, which is why it is p96 and last.
- `chat.html` is one of exactly three HTML explainers scheduled for the release
  - this one, Lane 2's approval flow, and Lane 8's architecture update. The
  epic records why: one per lane would be busywork.
- **Open:** whether the demo should also show a DELIVERY to two channels, which
  is `20260804-115319`'s claim and currently only has unit tests behind it. It
  would make the demo longer and would make the lane's proof complete. Leaning
  yes; decide at PLANNING when the demo is scoped.
- **Open:** how much of `chat.html` is prose versus a diagram. The operator
  asked for ASCII diagrams over prose during the lane cut, and that preference
  should carry into the explainer.
