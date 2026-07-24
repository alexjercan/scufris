# Retro: Session ownership index + multi-session history; drive the switcher from it

- TASK: 20260724-111947
- BRANCH: fix/session-ownership-index (squash-landed 236c129)
- REVIEW ROUNDS: 1 (out-of-context APPROVE, one NIT fixed same round)

Process only; TASK.md has the what/why/evidence, NOTES.md the design/fix record,
DECISION.md the ownership model.

## What went well

- **Spike -> plan -> decision paid off.** The leak's root cause (ownership
  inferred from a disk scan, not recorded) and the forward-only no-backfill call
  were settled in SPIKE/DECISION before any code, so /work was mechanical and the
  review had a written spec to judge against. Zero design churn.
- **Repro-first, pinned at the boundary.** The leak regression was written first
  and went red on master (`['sub-sess','orch-sess'] == ['orch-sess']`); an A/B
  sabotage re-reddened it. The out-of-context reviewer independently re-derived
  the red by inspection and agreed.
- **Applied a prior lesson on purpose.** `api-preserving-refactor-still-drops-an-old-contract`
  (from 20260723-001251, the sibling registry task) was in mind: this widened the
  registry's value shape, so I proactively added tests pinning the NEW mechanism
  (multi-session history, legacy-shape load) rather than leaning on the unchanged
  existing suite. The reviewer confirmed no old contract silently retired.
- **Registry already existed as the single home of session ids** (decision
  20260723-001251), so this was a value-shape widening, not a new store - small,
  low-risk diff.

## What went wrong

- **A DoD mapped a claim to a test that did not exercise it.** The DoD named
  `test_orchestrator_switcher_lists_registry_history` as proving "newest first,"
  but the test asserted only set membership, and both fixtures shared an mtime -
  so the sort key + `reverse=True` were untested. Root cause: at plan time I
  wrote the DoD-to-test mapping from the test's PURPOSE, not from what it would
  actually assert; the ordering claim needed distinct mtimes to be checkable and
  I did not encode that. Caught by the out-of-context review (R1.1 NIT), fixed by
  giving the rollouts distinct mtimes and asserting list order (A/B: red without
  `reverse=True`).
- **mypy `list`-shadow detour.** `AgentStore.list()` shadows the builtin `list`
  in class-scope annotations, so `-> list[str]` on a new method resolved to the
  method, not the type ("not valid as a type", then "not iterable" at the call
  site). A small detour to diagnose; fixed with a module-level `SessionIdList`
  alias. Seemed right to just write `list[str]` because every other method does -
  but they predate the `list` method textually.

## What to improve next time

- When a DoD item says "(test: X)", make X assert the SPECIFIC claim, not a
  neighbouring one; if the claim is ordering/quantity, the fixture must make it
  distinguishable (distinct mtimes/timestamps) or the assertion proves nothing.
- In a class that defines a method named after a builtin (`list`, `dict`, `id`,
  `type`), annotate returns of that builtin via a module-level alias bound
  outside the class.

## Action items

- [x] Recorded `dod-proof-must-exercise-the-named-claim` and
      `class-method-shadows-builtin-in-annotations` in LESSONS.md.
- No follow-up code tasks: parts 2 (20260724-111955) and 3 (20260724-111959)
  already exist and depend on this; part 2 is next.
