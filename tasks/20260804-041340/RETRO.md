# Retro: Fix the examples the package carve broke

- TASK: 20260804-041340
- BRANCH: fix/examples-carve-paths
- REVIEW ROUNDS: 1

## What went well

Planning measured instead of assuming. The record was written believing one
defect broke both named examples. Running all thirteen examples under a freshly
synced env, with the path corrected out of band, split the symptom into two
independent defects and matched every one of the thirteen against a predicted
cause. That measurement is what made the scope call in DECISION.md defensible
rather than a guess, and it is why the implementation found no surprises: the
reds reproduced exactly as predicted and the post-fix sweep printed exactly the
three predicted names.

Planning also falsified a Story premise before it cost anything. The Story said
`nix flake check` does not run examples; it does (`flake.nix:262` runs pytest,
`tests/test_examples.py` runs each `OFFLINE` entry as a subprocess). That turned
Step 3 from "invent a check" into "add one line to the tuple that already
exists", and it avoided building a second gate that could disagree with the
first.

Refusing to fold in the event-loop fix kept the diff at three code lines. The
alternative was importing another filed task's whole app-startup redesign and
two more examples this task never scoped.

## What went wrong

Nothing in the code. The single review finding (R1.1, MINOR) is a records
problem: `NOTES.md` still carries the pre-DECISION scope - "all thirteen exit 0,
and the four join the opt-in list" - and an illustrative `OFFLINE` block marking
three extra examples `# + new`. DECISION.md then narrowed the task to one entry
and the branch landed one.

The failed decision, and why it looked sound: NOTES.md was committed at PLANNING
(`3ae0aa7`) as the honest state of the analysis at that moment, and the
convention is that later records supersede earlier ones rather than rewriting
them. That convention is right for reasoning history. It fails for a record's
*forward-looking* claims, which do not read as history - a cold reader landing
on NOTES.md after the merge concludes the task under-delivered. The DECISION
that superseded it lives in a different file and does not announce itself from
where the stale claim sits.

## What to improve next time

When a DECISION narrows scope that an earlier NOTES.md already promised, add a
one-line pointer at the superseded paragraph in the same commit that accepts the
DECISION. One line, not a rewrite: the history stays intact and the stale claim
stops reading as a live promise.

Diagnosis of the three standard questions:

- **Breadth.** Not applicable in the usual direction - the diff is three code
  lines. It stayed that small because a filed sibling task
  (`20260803-014210`) already owned the second defect, and the boundary held.
- **Churn.** Zero review rework. No plan-time question would have prevented the
  one finding, because it is not about the plan's design; it is about a record
  hygiene step that runs *after* a DECISION is accepted.
- **Context.** No observed pressure. No checkpoint, compaction warning,
  delegation for context reasons, or handoff. The one delegation was the
  mandatory round-1 out-of-context reviewer, not a pressure response.

## Action items

- [x] R1.1 stands as a MINOR with an open Response; it does not block landing.
      Recorded here rather than fixed, since editing NOTES.md post-hoc is the
      exact rewrite this retro argues against - the pointer belongs in the
      DECISION commit, and that commit is already history.
- [ ] `20260804-053002` Done Means 3
      (`test_every_package_has_a_gated_example`) is the durable fix for the
      root cause. `OFFLINE` is a hand-written opt-in tuple: the gate existed
      and the entry did not, which is the failure mode this shape invites.
      Until that lands, every new example depends on someone remembering the
      tuple. No new task filed - it is already owned.
- [ ] Central knowledge writes succeeded and `knowledge check` exits 0:
      occurrences on `verification/a-green-gate-has-a-bounded-claim` and
      `verification/reproduce-before-explaining`, plus a new
      `changes/a-runner-built-path-hides-a-moved-module`. They are left
      UNCOMMITTED in `/home/alex/personal/agent-knowledge`, matching that
      repo's existing state - roughly fifty other lessons from prior tasks sit
      uncommitted there. A first attempt committed all of them together; that
      commit was reset and the working tree restored, because sweeping other
      tasks' unreviewed lessons into one commit is not this task's call.
- [ ] `20260803-014210` now carries the Notes line saying
      `examples/telegram_approval.py`'s path is correct, so enrolling it in
      `OFFLINE` is the last step of that task, not a rediscovery.

## Landing message

```
fix: point the examples at the carved hostd test path

The hostd carve (6d998c8) moved tests/test_host_actions.py to
packages/hostd/tests/, but examples/host_agent.py and
examples/telegram_approval.py still inserted ROOT / "tests" on sys.path and
imported host_files / host_runner from it, so both died at import. Point both
at the new directory.

host_agent.py joins the OFFLINE tuple in tests/test_examples.py, so the next
carve that breaks it fails the suite. telegram_approval.py now reaches
create_app and is red only on the event-loop guard that 20260803-014210 owns;
that task enrolls it when it lands.
```
