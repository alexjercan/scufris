# Retro: EPIC: Carve the code into a uv workspace of per-service packages

- TASK: 20260803-213242
- BRANCH: master (no epic branch; the six children each landed through their own)
- REVIEW ROUNDS: 2

## What went well

- **The children were the unit of landing, not the epic.** Five carves plus a
  deletion landed as six independent commits on `master`
  (`e7cb027`, `09cf946`, `6d998c8`, `705da31`, `e2c3bbb`, `0879f2d`), each green
  on its own. A 10k-line tree reshape never existed as one reviewable diff, and
  a failure would have named the child that caused it.
- **Sequencing `host` before `hostd` was decided from the imports, not taste.**
  Six `hostd` modules already imported `scufris.host.run`; the understanding
  pass read that and re-ordered the children rather than discovering it mid-move
  (`06c68f2`).
- **`hostctl` was flagged in advance as the one non-move.** The record said to
  expect real edits - `EventBus` and the generic `Supervisor` hoisted, `Settings`
  narrowed - so the child that ran it was planned as work, not as `git mv`.
- **The allowlist beat the property check.** `CORE_MODULES` was chosen over
  "declares no `__tablename__`" precisely because the growth already planned for
  `core` would have satisfied a property check trivially. That call held: `core`
  grew from three modules to five, and each addition cost an allowlist edit and
  a written justification instead of passing silently.
- **Both review rounds ran out-of-context.** Round 1's eight findings were all
  things the implementing context believed it had already said correctly.

## What went wrong

- **Two Done Means had no child that owned them.** Done Means 3 (the declared
  graph) and 6 (the example gate) were written into the epic and then not
  assigned to any of the five carve children, so they sat unproven until the
  epic was already in WORKING and had to be seeded as `20260804-053002`
  (`c240f08`). It seemed sound at plan time because the carve children were
  scoped by DIRECTORY - each one moved a package - and a cross-cutting proof
  belongs to no single directory, so it fell between all five.
- **The epic record went stale against its own amendment.** `20260803-214749`
  hoisted `EventBus` and `Supervisor` into `core` and recorded the retraction in
  this task's `DECISION.md`, but the epic's Story, `AGENTS.md:18` and the `core`
  facade docstring all kept describing the three-module `core` that was planned.
  Round 1 spent five of its eight findings (R1.1-R1.5, R1.8) on that drift. The
  child amended the decision and did not amend the places the decision was
  QUOTED.
- **The examples broke silently in the carve and were caught outside it.**
  `20260804-041340` exists because moving code out from under the example paths
  was not part of any child's done criteria.
- **Guard tests written green on a clean tree proved nothing.** R1.6: the facade
  rule scanned only `packages/*/src/*` and `scufris/`, exempting the whole suite
  from the rule the suite exists to enforce - and two live violations were
  already sitting in it. R1.7: three `assert roots` anti-vacuity guards could
  not fail, because `_import_roots()` unconditionally inserts `"scufris"`.
- **The same vacuity recurred inside its own fix, twice.** Extending the scan to
  tests left the new arm silently green when deleted, which the fix caught and
  paired with a falsifier. Round 2 then found the layer under that: neutering
  the tests map at the real CALL SITE (`_facade_problems(roots, {})`) still
  leaves all six tests in the file green (R2.1, NIT). A falsifier that drives
  the helper does not prove the production call passes the helper real inputs.
- **`nix flake check` was red before the review round on the records check.**
  `REVIEW.md` had been written as prose with none of the schema lines, so Done
  Means 9 first went green in round 1's fix rather than in the work phase.

## What to improve next time

- **Assign every Done Means to a child before the epic leaves PLANNING.** A
  proof no child owns is a proof nobody writes. When children are scoped by
  directory, a cross-cutting proof needs its own child by construction.
- **A child that amends a parent's decision edits every place the parent states
  it, in the same commit.** `DECISION.md` is the record of the change, not its
  propagation. Grep the claim, not just the decision file.
- **A move child's done criteria must include the callers of what moved** -
  examples, entry points, and packaging outputs - or the breakage surfaces as a
  separate task after the fact.
- **Pair a checker with a falsifier at the moment it is EXTENDED, not only when
  it is written**, and drive the falsifier through the PRODUCTION call, not only
  through the helper. This file's own convention caught the second layer; round
  2 found the third.
- **Run `nix flake check` before requesting review**, not after the first round.
  It covers the record schema, so a review record written by hand can fail a
  Done Means that has nothing to do with the code.

## Action items

- `20260804-112025` (filed this round): `_wait_state` (`tests/test_app.py:2577`)
  polls for 2s and then returns the last state instead of failing, so a slow
  machine reads a timeout as a state mismatch. Observed once as
  `test_orchestrator_chat_uses_server_cwd` failing in a full run and passing in
  isolation. Pre-existing; outside this epic's diff.
- R2.1 and R2.2 are open NITs in `REVIEW.md`, non-blocking, to be carried by the
  next touch of `tests/test_package_boundaries.py` and this record.
- The two `manual:` proofs - Done Means 10 and `20260803-214746`'s acceptance -
  are carried unconfirmed to this close. Neither was ever put to the maintainer;
  judge `core` against the five modules that shipped, not the three planned.
- `packages/telegram` stays uncarved until this epic's open question is answered:
  whether host approvals are conversation events. Recorded in the Story; the
  answer decides which package owns the approval card.

## Landing message

```
refactor(packaging): carve the code into a uv workspace of per-service packages

Five of the ten declared units ship: core, host, hostd, hostctl and the scufris
composition root, with web/ unchanged. Each is a distribution with its own
dependency list, and the boundary is enforced rather than claimed - a package
may import a sibling's public facade and never its models or repo.

Five tests hold the shape: the facade rule over source and each member's tests,
DECLARED_GRAPH compared for equality and acyclicity, the CORE_MODULES allowlist
that makes every addition to core cost a justification, model registration so no
package's tables can silently vanish from the migration metadata, and an example
gate proven to cover every member.

The legacy agent router, the JSON import path and the pre-v0.2.0 migration chain
are deleted. agents, chat, flow and telegram are deliberately not carved: the
first three hold code 20260729-102157 replaces, and telegram waits on whether
host approvals are conversation events.
```
