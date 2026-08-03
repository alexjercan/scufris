# Retro: Close the round-2 findings from the create_app assembly extraction

- TASK: 20260803-102351
- BRANCH: refactor/close-round2-findings
- REVIEW ROUNDS: 2

## What went well

- The three inherited findings closed exactly as scoped. The implementation
  diff stays inside `tests/test_agent_run_router.py`, `scufris/README.md` and
  `scufris/host_approval_bridge.py`; nothing grew.
- R2.1b's `cmd:` proof carries a `grep -q` guard because the trap test was
  already green on base. Without it the criterion would have been green before
  and after and proved nothing. Guarding a proof whose test predates the change
  is the reusable move.
- The new assertion cannot pass vacuously: `/fork` returns 200 only past its
  404, 409 and 422 arms, and deleting `ForkingRunService` breaks the test. The
  review re-derived both independently.
- The file-size guard fired during work rather than at review, and the response
  was to move the method, not to trim unrelated lines or allowlist the file.
  The ratchet held.

## What went wrong

- The close-out claimed all four `cmd:` proofs ran red on base. Proof 4 is
  `python -m pytest && ruff check . && mypy .`, a regression guard that is
  green on base by construction. Round 1 caught it as MAJOR.
  Root cause: the claim did not originate in the close-out. TASK.md's planning
  Notes already asserted "All four `cmd:` proofs run red on master before the
  change (exit 1 each)", and the close-out inherited it verbatim. A false
  planning claim that nothing re-checks survives into the very paragraph meant
  to substantiate it.
- The R2.1b fix falsified a docstring the diff itself depended on. The module
  docstring said "only the diagnostics fake is redefined"; adding
  `ForkingRunService` made that false, and DECISION.md was at that moment
  quoting the sentence as its precedent. Round 1 caught the docstring (MINOR);
  round 2 caught the DECISION.md quote that fix left stale (NIT, still open).
  Root cause: a sentence quoted in one record but owned by another file has two
  copies and no link between them.
- Both round-1 findings were record-accuracy defects. Neither is reachable by
  the suite, `ruff` or `mypy`, which is why review was the only thing that
  could find them.

## What to improve next time

- Breadth: not a factor. Three findings, three files, one implementation
  commit. No split was missed.
- Churn: `plan`'s proof triage is the lever. A Definition of Done that mixes
  targeted proofs with a whole-suite regression guard should label which is
  which at plan time, and the red-on-base assertion should be written per
  proof rather than as one sentence covering all of them. Had the plan's Notes
  said "proofs 1-3 red on base; proof 4 is a regression guard, green on both",
  the close-out would have had nothing false to inherit and round 1's MAJOR
  would not exist.
- A Step that edits a docstring, comment or README sentence quoted elsewhere in
  the task's own records should sweep those records in the same commit. The
  cheap check is `grep` for a distinctive clause of the old sentence across
  `tasks/<id>/` before ticking the Step.
- Context: no pressure observed. No checkpoint, handoff, delegation or
  compaction warning appears in any record for this task.

## Action items

- R2.1 (NIT) stays open: `tasks/20260803-102351/DECISION.md:45` quotes the
  pre-amendment docstring sentence. Not blocking, and carried as history the
  same way this task's own source findings were.
- Splitting `tests/test_orchestrator_routers.py` along its three rigs (project,
  agent-run, agent-record) is filed as task 20260803-191946. The 897/900
  headroom against `scripts/check_file_size.py` is load-bearing - the next
  addition to any of those rigs hits the wall this task hit.

## Knowledge

Written to `/home/alex/personal/agent-knowledge` (project=scufris), all three
accepted and `knowledge check` clean:

- `planning/label-the-proof-that-cannot-go-red` - new. A Definition of Done
  mixes targeted proofs with whole-suite regression guards; label which is
  which at plan time, because one blanket "all proofs run red on base" claim
  is false for the guards and gets copied forward into the close-out.
- `testing/prove-the-test-can-fail` - occurrence added. When the change adds an
  assertion to a test that already passes, conjoin a `grep -q` for the new
  assertion so the criterion observes the change rather than the file's prior
  state.
- `docs/update-restatements-with-the-source` - occurrence added. Task records
  count as restatements while the task is open: a DECISION.md that quotes a
  docstring to justify a choice goes stale the moment a later round edits the
  quoted line.

## Landing message

```
refactor(tests): drive /fork in the agent-run trap and clear two stale symbols

Close the three MINOR/NIT findings the router extraction's APPROVE left open.
The agent-run booby-trap now drives /fork under the same four __init__ traps,
covering 15 of 16 routes; /events stays out because it relays a live bus that
nothing closes. The fake providing fork_seed lands as ForkingRunService in the
test that needs it, because the shared rig file is at its line cap.

scufris/README.md's trust-boundary table now names
telegram/wiring.py::build_approval_ops instead of the app symbol an earlier
task deleted, and host_approval_bridge.py drops a module logger it never used.
```
