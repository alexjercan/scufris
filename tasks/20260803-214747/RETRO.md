# Retro: Move the root helper into packages/hostd

- TASK: 20260803-214747
- BRANCH: refactor/hostd-package
- REVIEW ROUNDS: 2

## What went well

- Scratch-verifying the drift guard BEFORE writing the plan changed the plan.
  A deliberately mismatched `scufris-hostd==0.2.0` resolved and synced clean,
  which killed the DoD's original `importlib.metadata` comparison and produced
  a file-based test instead. That experiment cost minutes and prevented a guard
  that would have passed while guarding nothing.
- Checking the carve at its real boundary. `hostd`'s contract is a unix socket,
  not an import rule, so the task added `examples/hostd_socket_roundtrip.py`
  (raw `AF_UNIX`, offline) on top of the existing VM test. Round 1 found no
  behavior finding in an 1147-line diff.
- Running the helper package SECOND. The reorder recorded in the Story turned
  NOTES.md open question 1 into a non-issue: `scufris/hostd/` already imported
  only `scufris_host` and `scufris_core` by the time this task started.

## What went wrong

- Every one of the eleven findings across both rounds was a record or
  documentation defect. Zero were code. Three of them - the CHANGELOG stating
  the breaking case backwards, the README options-table default, the three
  relative links the `git mv` broke - share one root cause: the plan enumerated
  the SOURCE files that name the moving tree and the doc files that reference
  it by path, but never asked which prose makes a claim that the move itself
  falsifies. A link is a path reference and got swept; "a deployment pinning
  `package` by hand needs no change" is a claim about the diff, and nothing in
  the plan was looking for those.
- The DoD named a nix attribute that does not exist
  (`checks.x86_64-linux.scufris-hostd-vm-test`). It was written from the shape
  of the other checks rather than from `nix eval`, so `tatr proofs` emitted an
  unrunnable command until R1.4. The close-out disclosed it rather than hiding
  it, which is why it stayed a MINOR.
- The R1.6 fix replaced a wrong count with a wrong gloss, and round 2 caught it
  as R2.1. Guessing at what a tool prints, twice, when the tool was one command
  away. The final text quotes only the two strings the run actually emits.
- Two mid-implementation restructures (DECISION 5a, 5b) the plan did not
  anticipate: a second test module imported the moving test module, and
  `tests/test_domain_routers.py` sat one line under a 900-line cap that a
  four-imports-into-one collapse pushed over. Both are recorded and the moves
  are verbatim, so neither is a design failure - but both are the same shape,
  a file sitting at a ratchet where any edit at all trips it.

## What to improve next time

- Breadth: the diff is large (66 files, 1147 insertions) and correctly so - a
  distribution carve touches every importer at once and is not independently
  landable in halves. No missed split. The only unplanned growth was 5b's
  329-line test extraction, forced by a cap.
- Churn: the plan-time question that would have prevented five of the eleven
  findings is not in `plan` today. It is: "which sentence in the docs becomes
  FALSE when this lands, and which relative link changes depth?" A path grep
  finds neither. For any `git mv` of a directory containing a README, re-check
  link depth mechanically; for any change to a packaged output, re-read the
  operator-facing prose as an operator.
- Verify a proof command by RUNNING it at plan time rather than pattern-matching
  a neighbouring one. Same for any evidence line that quotes tool output.
- Context: no pressure observed. No compaction warning, no checkpoint, no
  handoff, one worktree throughout. Both review rounds used out-of-context
  subagents.

## Action items

- Sweep the remaining `packages/*/src/*/README.md` link depths in the next
  carve task. 09cf946 left one broken link that this task repaired incidentally
  (R1.3), which suggests the next carve will leave one too.
- The `tests/test_domain_routers.py` split (DECISION 5b) is the second file to
  hit the 900-line cap during a mechanical edit. If a third appears, revisit
  the cap rather than the file.

## Landing message

```
refactor(hostd): carve the privileged helper into packages/hostd

Move scufris/hostd/ to packages/hostd/src/scufris_hostd as the
scufris-hostd distribution, with its console script, its three test
modules and its README. Nothing an operator sees changes: same unit,
same socket, same PROTOCOL_VERSION = 1, same audit lines.

The twelve root modules that reached into actions, audit, protocol and
executor now go through the facade, which gains encode. The same-wheel
guarantee that kept the two halves of the socket protocol in step is
replaced by an exact scufris-hostd== pin plus
test_the_app_pins_hostd_to_one_exact_version, because uv drops the
specifier for a workspace source and no other gate here would notice it
rotting.

flake.nix gains packages.scufris-hostd and the NixOS module defaults to
it, so a deployment that pinned package to packages.scufris must
re-point it. examples/hostd_socket_roundtrip.py drives propose ->
preview -> approve -> apply -> audit over a raw unix socket, offline.
```
