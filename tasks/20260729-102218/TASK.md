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
- BLOCKING QUESTION (2026-07-29 backlog review), answer it FIRST in the spike:
  who actually retrieves? Scufris owns no search or fetch capability - there is
  no web tooling anywhere in `scufris/`. As written, this epic inherits whatever
  the backend CLI happens to have: codex and claude can search the web, but the
  opencode backend drives a local llama.cpp server with no web access at all, so
  a swarm on that backend produces citation-shaped hallucination. Decide whether
  Scufris owns retrieval (a search/fetch MCP with its own provenance) or
  research runs are restricted to backends that can retrieve. Everything else in
  this epic - the evidence ledger, the skeptic, the provenance UI - is
  unimplementable until that is settled, since the ledger's excerpt and hash
  fields presuppose a fetch Scufris can see.
- The differentiated value here is the LEDGER and its provenance, not the
  fan-out; codex and claude already run multi-step research on their own.

## Manual Acceptance

- (pending) 20260729-102222: the final report makes it easy to inspect sources,
  conflicting evidence, and how the conclusion was reached.

## Sequencing

- Post-v0.1.0 order (2026-07-29 backlog review): LAST of the five backlog
  epics, and blocked until the retrieval-ownership question above is answered.
  It is also the epic most at risk of rebuilding what the backend CLIs already
  do, so it should be scheduled only when the ledger and provenance are wanted
  for a real body of research.
- Stays `backlog` at priority 0 until pulled into a release plan.

## Flow State

- FLOW STEP: PLANNING
