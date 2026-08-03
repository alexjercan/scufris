# Retro: Delegate legacy /api/agent/* routes to orchestrator diagnostics

- TASK: 20260801-100415
- BRANCH: fix/legacy-agent-diagnostics
- REVIEW ROUNDS: 2

## What went well

- The plan's line references (`scufris/app.py:3571-3603`, `:1881`, `:1890`,
  `:3430`, `:2875-2884`, `:176-178`) were all accurate at implementation time,
  so the production move was mechanical.
- DECISION.md's surface table made the fourth consumer findable: removing the
  last `agent_health` import surfaced the digest `health()`, and the table was
  extended rather than the omission argued away.

## What went wrong

- Breadth: the diff is 623 lines, but only 104 of them are `scufris/app.py`.
  The growth is test adaptation, not scope creep or a missed split - the
  behaviour change (envelope shape plus short-circuit removal) invalidated
  existing tests wholesale. The one out-of-plan production edit (the digest
  `health()`) was not independently landable: leaving it would have kept
  `agent_health` imported for a stale reading.
- Churn: both load-bearing round-1 findings (R1.1, R1.3) and both open round-2
  findings (R2.1, R2.2) are the same defect - a test that passes with OR
  without the change it guards. DECISION-4 ("the `agent_enabled` short-circuit
  goes") and the frontend unwrap were each asserted by a test that could not go
  red. The plan-time question that would have caught it is the cold-reader
  rationale test in `plan/decision.md`: a decision that DELETES a guard has to
  name the observation that distinguishes the two worlds, and neither decision
  did.
- Round 1's verification prose recorded 982 passing tests after its own R1.3
  fix deleted one; Round 2 corrected it to 981 against the Close-out. A count
  copied forward across a round is not re-derived evidence.

## What to improve next time

- When a decision removes a short-circuit or guard, write the red-proof into
  the Step: state the input under which the old and new code DIVERGE. An empty
  fixture makes them agree, which is how R2.1 survived R1.3's fix.
- When a short-circuit is deleted, grep for default-`Settings` construction in
  the affected tests. Three disabled-agent tests started reading the
  developer's real `~/.codex` the moment the guard went.
- Context: the only observed pressure was the review being delegated
  out-of-context for both rounds, which worked; no compaction or checkpoint was
  recorded. Nothing to split next time.

## Action items

- 20260803-034922 - pin the two tests Round 2 left open (R2.1, R2.2). Both are
  non-blocking under the APPROVE; neither guards production behaviour that is
  wrong today.
- 20260803-032950 - `scufris/health.py:258` still reads a CODEX session count
  for a claude or opencode orchestrator, on both the legacy and scoped health
  surface. Consistent between the two, so not a regression here.
