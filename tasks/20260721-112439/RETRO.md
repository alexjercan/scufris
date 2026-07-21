# Retro: B5a reserved orchestrator agent record

- TASK: 20260721-112439
- BRANCH: feature/orchestrator-reserved-record
- REVIEW ROUNDS: 1 (out-of-context APPROVE, 1 NIT addressed)

## What went well

- The out-of-context RECON before touching code was the highest-leverage move:
  it revealed B5 conflates ~4 architectural changes (retire Agent protocol,
  unify sessions, converge chat, retire exec), which turned a risky mega-change
  into an approved 5-slice plan. Surfacing the re-cut to the user (who chose the
  full split) beat grinding a 2000-line change.
- Explicit SCOPE GUARDS in the B5a plan ("does NOT retire the Agent protocol";
  "two chat paths coexist, temporary") kept the slice small and let the reviewer
  verify it stayed in its lane.
- The cleanest call was DEFERRING editable orchestrator config to B5b instead of
  shimming a settings-persistence path into B5a. B5a has no settings-persistence
  seam of its own; wiring one would have been throwaway once B5b unifies. update
  -> 409 now, honestly recorded.
- Holding the orchestrator run-state in two in-memory fields (single active
  session, reset on restart) gave it a WORKING per-agent chat immediately,
  without polluting agents.json, and defers real session storage to B5c where it
  belongs. A small, honest increment.

## What went wrong

- Adding an always-present synthetic record broke 4 tests that asserted an EMPTY
  agent list. Expected fallout, caught by running the suite - but a reminder that
  "the list is never empty now" ripples to every empty-list assertion (and the
  mcp `_list_agents_text` no-agents branch, which became dead code, R1.1 NIT).

## What to improve next time

- When introducing an ALWAYS-PRESENT synthetic item into a collection, grep for
  every "empty" / "== []" / "no X" assertion + affordance up front (list tests,
  empty-state UI, the "none configured" branch) - the invariant "this collection
  is never empty" invalidates all of them at once.
- Recon-then-recut is now a proven pattern for a task that turns out to be an
  architectural umbrella: buy the map, split with scope guards, land the safe
  slice first. Ledgered.

## Action items

- [x] Review APPROVE, NIT addressed.
- Next: B5b (retire the Agent protocol -> orchestrator via get_backend). The
  HIGH-RISK slice; B5a laid the projectless/get_backend seam it builds on.
