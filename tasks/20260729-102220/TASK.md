# Add the research run and evidence ledger model

- PRIORITY: 0
- TAGS: feature, backlog, research, agents, backend
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Story

As a research operator, I want research plans, questions, sources, evidence,
claims, conflicts, budgets, and synthesis artifacts stored as durable typed
records, so that a run can be resumed, audited, and independently checked.

## Steps

- [ ] Implement versioned schemas for research request/plan, work item, source,
      evidence, claim, citation, conflict, quality check, budget, and result.
- [ ] Persist the research graph transactionally with stable IDs, ordering,
      restart recovery, pagination, retention, and links to agents/runs/
      artifacts.
- [ ] Deduplicate sources and evidence by canonical reference and content hash
      while retaining each researcher's attribution and independent assessment.
- [ ] Track claim-to-evidence support/contradiction edges and prevent citations
      from referring to absent or changed sources.
- [ ] Add APIs for plan, progress, evidence, conflicts, budgets, cancellation,
      and final artifacts with project/run authorization.
- [ ] Add fixtures for duplicate sources, conflicting claims, source updates,
      inaccessible evidence, partial agents, exhausted budgets, and resume.
- [ ] Add a deterministic example that constructs and validates a small
      evidence ledger without a live provider.

## Definition of Done

- A concurrent multi-researcher fixture survives restart without losing or
  duplicating evidence (test: `test_research_evidence_ledger_provenance`).
- Every material claim can enumerate supporting, contradicting, and missing
  evidence (test: `test_research_claim_evidence_graph`).
- Changed or missing sources invalidate affected citation anchors explicitly
  (test: `test_research_citation_integrity`).
- The deterministic example passes
  (cmd: `python examples/research_evidence_ledger.py`).

## Notes

- Epic: 20260729-102218.
- Depends on: 20260729-102219, 20260729-102147, and 20260729-102203.
- Reuse the general run and artifact identity model instead of creating a
  separate research-only observability system.
