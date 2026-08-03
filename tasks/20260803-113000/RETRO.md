# Retro: Prove the startup sweep clears a building row orphaned by a crash

- TASK: 20260803-113000
- BRANCH: test/orphaned-building-row-swept
- REVIEW ROUNDS: 1

## What went well

- The mechanism was probed on the base during planning - the row's state after a
  clean teardown, the reachability of `open_database(tmp_path)`, and the
  falsification with `abandon_builds()` removed - so the Steps described a
  verified path rather than an intended one. Implementation was one pass, no
  rework, one MINOR finding.
- The plan's own premise turned out to be wrong mid-planning (the lifespan does
  not close the config supervisor; the portal teardown cancels the build), and
  the correction was written into Notes with the conclusion re-derived instead
  of silently kept. The conclusion survived; the stated reason changed.
- The test asserts `state is ChangeState.CANCELLED` before forcing the row back
  to BUILDING, so the reason the direct store write exists is itself a proof.
  A comment claiming "a clean shutdown cannot produce this" would have rotted
  the first time the shutdown path changed.
- Breadth: one import line and one test function. No split was missed; the
  live-process test was deliberately left byte-identical rather than widened,
  which kept each test about one mechanism.

## What went wrong

- Step 2 instructed the code comment to name `20260803-113000` and
  `20260803-014401 DECISION.md 1`. `AGENTS.md:103` forbids task IDs in code
  comments and the policy table says to keep the invariant as a fact about the
  code and delete the lore. The plan encoded a repository-policy violation, and
  the implementation followed it faithfully. R1.1 (MINOR, open).
- Why it looked sound: the immediate neighbour
  (`tests/test_nixos_config_change.py:632`) and the production docstring for the
  code under test (`scufris/hostconfig/changes.py:158`) both carry the same ID
  references. Local precedent read as the convention; the written policy was not
  re-checked while drafting a Step that dictated comment text.
- Churn: none from review. The one finding came from the AGENTS.md read, not
  from the from-scratch challenge or the cold-reader test - the design was never
  in dispute.

## What to improve next time

- A plan Step that dictates the *content* of a comment or docstring has to be
  checked against the repository comment policy before it is written down.
  Precedent in the neighbouring lines is not evidence of the policy; it is often
  evidence of the same mistake made earlier.
- When a plan's stated mechanism is corrected mid-planning, keep asserting the
  conclusion in the test rather than restating it in prose. That is what made
  the correction here cheap.

## Action items

- R1.1 stays open and non-blocking. The ID references it names exist in at least
  three places (this test, the neighbouring test, and `abandon_builds`'s
  docstring), so fixing only the new one would leave the file inconsistent.
  Either sweep them in one task or amend `AGENTS.md` to match the practice; that
  decision is wider than this diff and is not taken here.

## Landing message

```
test(hostconfig): prove the startup sweep clears a crash-orphaned building row

The existing restart test never exits its first TestClient, so the row it
sweeps still belongs to a live process. Add a test that lets the first process
end, asserts a clean teardown leaves `cancelled`, re-establishes the `building`
row with an empty error through ConfigChangeStore - the state a SIGKILL leaves
- and asserts the restarted app fails it with a restart reason and no proposal.

Tests only; abandon_builds is unchanged.
```
