# Retro: Backend run-one-tool endpoint + param schema for the 'try it' runner

- TASK: 20260720-134545
- BRANCH: feature/tool-run-endpoint
- REVIEW ROUNDS: 1 (APPROVE)

## What went well

- Probed the dependency before designing: ran `mcp.call_tool` live to learn its
  real return shape (a 2-tuple, not the annotated `Sequence | dict`) and its error
  behavior (`ToolError` for unknown/bad/missing args) BEFORE writing the endpoint.
  The plan's "verify live" facts were load-bearing and correct.
- One-round APPROVE with only MINOR/NIT. The out-of-context reviewer independently
  re-ran the whole suite and verified the two non-obvious claims (routing non-change,
  the in-process disabled_tools gate); the in-session pass re-derived the gate claim
  (`apply_disabled_tools` runs only in the codex subprocess `main()`), so the APPROVE
  rests on confirmation, not trust.
- The ledger paid off in advance: `type-change-fails-strict-tsc-not-vitest` and the
  new-required-field lesson were recalled at plan time, so adding `parameters` to the
  shared `AgentTool` interface came WITH the three fixture fixes in the same pass -
  no red frontend discovered later.

## What went wrong

- I verified the frontend during `/work` with `vitest` + `eslint` + `prettier` but
  NOT `npm run build` (the webpack ts-loader gate) - the exact gap
  `type-change-fails-strict-tsc-not-vitest` warns about. It happened to be clean
  (all three `AgentTool` constructors were fixed), but I only confirmed the real
  type gate AFTER landing, during this retro. Root cause: ran the tests I reached
  for by habit, not the gate the ledger names for this specific change class.
- The plan's step "add the run route to the write/read classification" baked in an
  assumption that a write-auth gate existed. It did not (`_route_tags` is OpenAPI
  tags only; no auth gate over these routes). Cheap because I read the routing
  before editing, but the step should have been phrased verify-first.

## What to improve next time

- When a change adds/removes a field on a shared TS interface, run `npm run build`
  (or `npm run ci`) as part of verify, not just `vitest` - vitest transpiles and
  does not type-check. This is the third occurrence; see the ledger promotion.
- Phrase plan steps that assert a gate/mechanism exists ("add it to the X
  classification") as verify-first ("confirm whether X gates these routes, then...").

## Action items

- [x] Ran `npm run build` on master post-land: compiles clean, no breakage landed.
- [x] Bumped `type-change-fails-strict-tsc-not-vitest` to x3 and moved it to Pending
      promotions (proposed: an AGENTS.md verify-step line).
- [x] Added ledger lesson `trust-runtime-shape-over-annotation`.
- No follow-up code task: the frontend consumer is already planned as 20260722-213000.
