# Spike: define the evidence-backed research swarm workflow

- PRIORITY: 0
- TAGS: spike, backlog, research, agents
- KIND: SPIKE
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Story

As a product designer, I want a precise research-swarm contract, so that
parallel agents improve coverage and verification without multiplying cost,
duplicating sources, or hiding unsupported conclusions.

## Steps

- [ ] Define representative research jobs: quick fact check, technical survey,
      product comparison, literature review, and open-ended investigation.
- [ ] Define stages and responsibilities for planner, parallel researchers,
      evidence deduplication, skeptic/claim checker, and synthesizer.
- [ ] Define source admission, provenance, citation anchoring, quoting limits,
      inaccessible sources, contradiction, freshness, and confidence rules.
- [ ] Define fan-out, per-agent/source/time/token budgets, cancellation,
      partial-failure behavior, retry, and stopping rules.
- [ ] Compare fixed workflow orchestration with dynamic recursive spawning and
      choose where bounded adaptation is allowed.
- [ ] Define measurable quality checks and a single-agent baseline so a swarm
      is used only when it improves the job.
- [ ] Produce example research plans and evidence ledgers, write `SPIKE.md`,
      record the selected workflow in `DECISION.md`, and refine child tasks.

## Definition of Done

- The spike covers all five job shapes and says when not to use a swarm
  (cmd: `rg -n "fact check|technical survey|product comparison|literature review|open-ended|single-agent" tasks/20260729-102219/SPIKE.md`).
- Stage contracts, evidence/citation schema, budgets, stopping rules, and
  partial-failure policy are decided
  (cmd: `test -f tasks/20260729-102219/SPIKE.md && test -f tasks/20260729-102219/DECISION.md && tatr check 20260729-102219`).
- At least one worked example includes conflicting evidence and an unsupported
  claim rejected before synthesis
  (cmd: `rg -n "conflict|unsupported|skeptic" tasks/20260729-102219/SPIKE.md`).
- The user accepts the balance between autonomy, cost, and evidence
  quality (manual: user check).

## Notes

- Epic: 20260729-102218.
- Depend conceptually on agent presets and artifacts, but keep the spike
  executable before those implementation epics finish.
- Prefer a bounded workflow over unrestricted recursive agent spawning.
