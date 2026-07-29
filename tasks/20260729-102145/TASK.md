# EPIC: Make Scufris durable and backend-truthful

- STATUS: OPEN
- PRIORITY: 0
- TAGS: goal,epic,backlog,reliability,backend

## Epic

Make Scufris safe under the concurrent writes produced by multiple agents and
make every operator surface report the backend that is actually running.
This epic addresses the two highest-risk findings from the 2026-07-29 project
audit before broader automation work builds on the current state layer.

## Done Means

1. Concurrent API mutations and simultaneous agent completions persist without
   exceptions, lost records, or corrupt state across a restart
   (test: `test_concurrent_state_mutations_survive_restart`).
2. Landing, agent settings, and Telegram report the same effective backend,
   model, health, tools, memory, and quota semantics for Codex, Claude,
   OpenCode, and mock agents
   (test: `test_orchestrator_surfaces_are_backend_consistent`).
3. The migration and recovery behavior is documented for an existing Scufris
   state directory (cmd: `rg -n "migration|backup|recovery" README.md scufris/`).
4. manual: with two agents completing at the same time, their sessions and
   outcomes remain visible after restarting Scufris.

## Child Tasks

- [ ] 20260729-102146 (p0, scufris) choose the transactional state persistence
      architecture
- [ ] 20260729-102147 (p0, scufris) migrate runtime state to concurrency-safe
      transactional persistence
- [ ] 20260729-102148 (p0, scufris) unify backend-aware orchestrator and
      Telegram diagnostics
- [ ] 20260729-103712 (p0, scufris) extract domain services and routers from
      application assembly

## Decisions

- Pending 20260729-102146 SPIKE.md and DECISION.md: persistence mechanism,
  migration boundary, and recovery policy.

## Manual Acceptance

- (pending) 20260729-102147: existing local state migrates without losing
  projects, agents, sessions, outcomes, or settings.
- (pending) 20260729-102148: backend/account information feels consistent
  across the landing page, agent settings, and Telegram.

## Flow State

- FLOW STEP: PLANNING
