# EPIC: Add evidence-backed research swarms

- STATUS: OPEN
- PRIORITY: 0
- TAGS: goal,epic,backlog,research,agents

## Epic

Add a structured research workflow that uses multiple agents only where
parallelism improves evidence coverage. A research swarm must produce a
traceable evidence ledger, citations, disagreements, budgets, and a synthesized
artifact rather than merely spawning several chats and concatenating answers.

## Done Means

1. A research request becomes a bounded plan with explicit questions, source
   policy, budgets, stopping rules, and expected output
   (test: `test_research_plan_requires_bounds_and_output_contract`).
2. Parallel researchers write deduplicated evidence and claims with source,
   retrieval, excerpt/hash, attribution, and confidence metadata
   (test: `test_research_evidence_ledger_provenance`).
3. A skeptic checks conflicts and unsupported claims before synthesis, and the
   final artifact links every material claim to evidence
   (test: `test_research_synthesis_rejects_unsupported_claims`).
4. The browser shows progress, evidence, disagreements, citations, budgets,
   cancellation, and export (test: `research-run.spec.ts`).
5. manual: a real research report is more trustworthy and useful than a single
   unstructured agent answer.

## Child Tasks

- [ ] 20260729-102219 (p0, scufris) define the evidence-backed research swarm
      workflow
- [ ] 20260729-102220 (p0, scufris) add the research run and evidence ledger
      model
- [ ] 20260729-102221 (p0, scufris) implement planner, researcher, skeptic, and
      synthesizer orchestration
- [ ] 20260729-102222 (p0, scufris) add research provenance, citations, and
      export UI

## Decisions

- Pending 20260729-102219 SPIKE.md and DECISION.md: research stages, source and
  citation contract, fan-out policy, budget/stop rules, and quality gates.

## Manual Acceptance

- (pending) 20260729-102222: the final report makes it easy to inspect sources,
  conflicting evidence, and how the conclusion was reached.

## Flow State

- FLOW STEP: PLANNING
