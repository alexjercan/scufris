# Retro: Record session ownership at launch per backend (part 2)

- TASK: 20260724-111955
- BRANCH: fix/session-launch-handles
- REVIEW ROUNDS: 1 (out-of-context APPROVE, one NIT fixed same round)

Process only; TASK.md has the what/why, NOTES.md the design/fix, DECISION.md the
scope call.

## What went well

- **Scope re-cut recorded, not silently done.** Part 1 changed the value/risk of
  part 2's spike bullets (the codex originator became a live regression risk vs
  `_SCUFRIS_ORIGINATORS` for zero payoff once listing went index-driven). I wrote
  a DECISION.md + GOAL.md deferral rather than quietly narrowing, so the dropped
  work is traceable into part 3. Under /flow this is the honest handling of "the
  plan's assumption changed since it was written."
- **Applied part 1's fresh lesson on purpose.** `dod-proof-must-exercise-the-named-claim`
  (added last cycle): the load-bearing bit here was the `StreamDone`
  substitution, so I A/B'd it (neuter the guard -> red) instead of trusting that
  the mint test covered it. The out-of-context reviewer re-ran the same A/B and
  agreed.
- **Self-contained seam.** Minting inside `ClaudeBackend.stream` (rather than
  threading a store-minted id through the supervisor) kept the diff small and
  still delivered "id known at launch, always on StreamDone."

## What went wrong

- **A caller and a pure helper recomputed the same predicate.** `stream` scanned
  disk to decide whether to mint, and `_claude_stream_args` scanned again to
  decide resume-vs-session-id - a redundant `rglob` per fresh turn. Root cause:
  I added the mint decision to `stream` without noticing the arg builder already
  derived the same resumability bit. Caught as the review's single NIT; fixed with
  an optional `resumable` override so the builder trusts the caller's decision but
  the pure-function unit tests still self-derive it.

## What to improve next time

- When a caller computes a predicate to make a decision and then calls a pure
  helper that needs the SAME predicate, pass it in (optional override) instead of
  letting both derive it - especially when deriving it touches the filesystem.

## Action items

- [x] Recorded no new ledger lesson (the double-scan is a one-off design smell,
      kept in this retro per the compound "one-offs stay in the retro" rule).
- Part 3 (20260724-111959) picks up the deferred codex/parent work; `_send` and
  the registry `parent_agent_id` field are the seams it extends.
