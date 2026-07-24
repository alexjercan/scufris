# Retro: parent-session routing for sub-agent escalations (part 3)

- TASK: 20260724-132830
- BRANCH: feature/parent-session-routing
- REVIEW ROUNDS: 1 (out-of-context APPROVE, one MINOR + one NIT fixed same round)

Process only; TASK.md has the what/why, NOTES.md the design, DECISION.md the
mechanism + alternatives.

## What went well

- **Stopping at the gate paid off big.** The seeded part-3 task inspected as a
  literal no-op (only the orchestrator spawns; `request_input` already reaches it;
  a bare `parent_agent_id` had no consumer). Instead of building dead plumbing, I
  surfaced that at the flow gate and asked the user for the scope. They chose the
  valuable version (route escalations to the spawning chat), which part 1's
  multi-session is what makes meaningful. The whole redefinition happened before a
  branch was cut.
- **Capture fell out for free.** The orchestrator's "current chat" is exactly the
  session its turn already resumes, so nothing had to be plumbed - only surfaced
  into the MCP env (`SCUFRIS_ORCH_SESSION_ID`), mirroring the existing
  `SCUFRIS_AGENT_ID` pattern.
- **Design call written down first.** filter-vs-annotate-vs-hard-filter and the
  fresh-turn edge went into DECISION.md before coding, so the filter boolean and
  the unattributed fallback were deliberate; the out-of-context reviewer re-derived
  the no-orphaning property and agreed.

## What went wrong

- **A user-facing rendering claim went unexercised by a test.** TASK step 5 and
  DECISION.md said `pending_agents()` "renders the parent chat in its table", but I
  only recorded the parent on the API row and tested the API - the tool's rendered
  table still showed ID/STATE/MESSAGE. The out-of-context reviewer caught the
  code/claim mismatch (R1.1). Root cause: I tested the data path and assumed the
  rendering followed, instead of asserting the operator-visible string. Same shape
  as last cycle's `dod-proof-must-exercise-the-named-claim` - the claim was about
  rendering, the test only covered the field.

## What to improve next time

- When a DoD/decision claims something USER-FACING is rendered (a table column, a
  chip, a message), the test must assert the rendered STRING, not just the
  underlying field. Data-present != displayed.

## Action items

- [x] Bumped `dod-proof-must-exercise-the-named-claim` to x2 in LESSONS.md with
      this rendering-vs-field occurrence.
- Seeded task 20260724-111959 is superseded by this; close it when this lands.
