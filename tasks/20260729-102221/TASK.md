# Implement planner researcher skeptic synthesizer orchestration

- STATUS: OPEN
- PRIORITY: 0
- TAGS: feature,backlog,research,agents,orchestrator
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT

## Story

As a user, I want Scufris to execute a research plan through bounded specialist
roles and quality gates, so that parallel work produces a supported synthesis
rather than several unchecked answers.

## Steps

- [ ] Add research presets for planner, researcher, skeptic/deduplicator, and
      synthesizer with least-privilege source and artifact capabilities.
- [ ] Implement the stage machine selected by 20260729-102219, including
      resumable transitions, fan-out limits, work assignment, and cancellation.
- [ ] Make researchers write structured evidence/claims rather than free-form
      final reports; isolate source budgets and failure state per work item.
- [ ] Run deterministic deduplication and skeptic checks for duplicate sources,
      contradictions, unsupported claims, citation integrity, and coverage.
- [ ] Permit synthesis only after configured quality thresholds or an explicit
      user override recorded in the audit trail.
- [ ] Enforce global/per-stage budgets and stopping rules; surface partial
      results when one researcher or provider fails.
- [ ] Add a mock source plugin and integration scenarios for success,
      contradiction, partial failure, cancellation, resume, and budget exhaust.
- [ ] Add an opt-in live-provider example for a small research question.

## Definition of Done

- A mock research run completes planner through synthesis with valid citations
  (test: `test_research_swarm_end_to_end`).
- Unsupported claims block synthesis unless an audited override is granted
  (test: `test_research_synthesis_rejects_unsupported_claims`).
- Partial failure, cancellation, resume, and budget exhaustion are deterministic
  and preserve collected evidence (test: `test_research_swarm_recovery_paths`).
- Agent concurrency never exceeds the approved plan limit
  (test: `test_research_swarm_enforces_fanout_budget`).
- The opt-in example is documented (cmd: `python examples/research_swarm.py --help`).

## Notes

- Epic: 20260729-102218.
- Depends on: 20260729-102220, 20260729-102206, 20260729-102207, and
  20260729-102919.
- Use the existing supervisor/backend abstraction. Do not introduce a new
  harness merely to implement workflow state.
