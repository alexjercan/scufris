# Retro: Split the agent runtime modules under the size cap

- TASK: 20260731-171428
- BRANCH: refactor/split-agent-runtime-modules
- REVIEW ROUNDS: 2

## What went well

One commit per split, each deleting its own `ALLOWLIST` entry, kept the guard
green at every commit instead of only at the tip - the reviewer checked that and
it held. Recording a green baseline (896 tests, ruff/mypy/filesize clean) before
touching anything made every later run a one-bit answer.

Both decide-once points were decided by MEASURING the file, not by guessing at
plan time, and the plan's two guesses were both wrong in opposite directions:
`agent/appserver.py` landed at 466 and did NOT need the `sandbox.py` the plan
anticipated, while `agent_store/store.py` landed at 667 and DID need its second
cut. Writing the threshold into the Step rather than the shape is what let that
happen without a re-plan.

The second cut turned out not to be a size trick: `awaiting_approval`,
`request_input` and `report_back` only ever built a `RunOutcome` and handed it
to `self._outcomes.set`, so they were `OutcomeStore` methods sitting on the wrong
class. The line cap surfaced a real ownership error.

## What went wrong

The close-out recorded a FALSE diagnosis and the review caught it (R1.1, MAJOR).
`ruff format --check` flagged the new `agent_store` package, and I attributed it
to the pre-split file being one of the 17 files the formatter would rewrite -
the plausible story, since 17 such files were already known from the baseline
run. Never measured. All four pre-split modules are format-clean at the
merge-base; the flagged lines were ones written during the split. The decision
seemed sound because a verbatim extraction "cannot" introduce formatting drift -
but the extraction was not fully verbatim, and the new delegating calls were
mine.

The `backends` facade re-exported `_context_from_status` with no consumer of that
path (R1.2). It was added to `__all__` to silence ruff F401 after the facade
imported it, which is backwards: the right fix for an unused import is to drop
the import.

## What to improve next time

Grep the string monkeypatch targets alongside the import sites during the
pre-split survey. `scufris.agent.STREAM_READ_LIMIT`,
`scufris.agent.shutil.which` and `scufris.backends._stream_app_server` are call
sites that no import-path grep reports, and they were the only thing in this
refactor that could have failed silently - a patch target that still RESOLVES
after a split but no longer reaches the global the code reads makes the test
pass while patching nothing.

When a check fires unexpectedly on moved code, run it against the PRE-move file
before writing down a cause. One command separates "inherited" from "I wrote
that", and the record is permanent either way.

Do not add a name to `__all__` to satisfy F401. Ask first whether anything
outside the package imports it through the facade.

## Action items

- None requiring a follow-up task. The two findings were record and facade
  hygiene, both fixed on this branch.

## Diagnosis

- **Breadth.** Large by design and correctly so: 3797 lines across four modules,
  landed as four independent commits in dependency order. No missed split - each
  commit is separately landable and separately green.
- **Churn.** One review round of rework, both findings self-inflicted at
  close-out/facade time rather than design time. The plan-time question that
  would have prevented R1.1 is not a `plan` question at all; it is the `work`
  rule "re-read edited artifacts, tool success does not prove correct content"
  applied to a RECORD rather than to code. R1.2 would have been caught by asking
  the cold-reader rationale test of the `__all__` comment: it named no consumer,
  which is exactly the tell.
- **Context.** One compaction mid-task, during split 4 of 4 (`agent_store`).
  The handoff cost nothing material - the three finished splits were already
  committed, so the recovery surface was one half-written package. Committing
  per split is what made the compaction cheap; a single end-of-task commit would
  have put all four splits in the uncommitted working tree at that moment.
