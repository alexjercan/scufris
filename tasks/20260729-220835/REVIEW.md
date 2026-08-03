# Review: Spike: define the actor-aware orchestrator conversation and flow-control model

- TASK: 20260729-220835
- BRANCH: master

## Round 1

- REVIEWER: maintainer (manual acceptance round, 2026-08-03)

The reviewable artifact of a spike is its reasoning and its mockup, not a diff.
This round is the acceptance round the DECISION's own gate paragraph named: the
maintainer played `tasks/20260729-220835/mockup.html` through the full scenario
at desktop and phone widths and read `SPIKE.md` and `DECISION.md` against it.

- VERDICT: APPROVE

Accepted as written:

- Option C. Scufris owns a semantic conversation; provider sessions are a cache
  keyed by `(conversation, backend, policy version)`.
- Four records, four owners, and the invariant that no projection becomes a
  second source.
- Typed actors, and the rule that only an `operator` event may satisfy a stop
  gate. An agent report is data, never an instruction.
- Workflow authority stays with tatr; Scufris asks `tatr flow -n`, requires the
  operator approval event, and returns a REASON on refusal.
- The tables the spike sketches - conversation, event, activity, delivery,
  assignment, run - were called out specifically as solid and simple enough to
  build on directly.

Two directions the round added, neither of which contradicts the record:

- [x] R1.1 (MINOR) tasks/20260729-220835/DECISION.md:1 - the model was scheduled
      against v0.3.0, behind a polish release. The maintainer cut that release:
      this becomes v0.2.0, implemented as a rewrite with NO backwards
      compatibility. The existing database is dropped rather than migrated and
      the legacy JSON import path is deleted outright, which removes the
      migration cost the "Paid" section anticipated. Recorded in the
      ratification paragraph; `tasks/20260801-154211/TASK.md` carries the
      schedule.
- [x] R1.2 (MINOR) tasks/20260729-220835/DECISION.md:1 - the round raised how
      the surviving code is PACKAGED: a `uv` workspace of per-service modules
      under one composition root, with host approvals as their own subproject
      and a shared library beneath. That is a packaging and dependency-direction
      question rather than a conversation-ownership one, so it is routed to its
      own record instead of being amended into this decision. Noted at the end
      of the ratification paragraph.

No BLOCKER or MAJOR findings. The spike's own "Open questions" - retention,
summary versioning, event granularity, re-seed eagerness, the
`SCUFRIS_ORCH_SESSION_ID` rename, and where the guard service lives - stay open
by design and are scoped to implementation tasks, not to this round.
