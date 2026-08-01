# EPIC: Make Scufris durable and backend-truthful

- STATUS: OPEN
- PRIORITY: 110
- TAGS: goal,epic,v0.2.0,reliability,backend
- KIND: EPIC
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT

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
4. All app-owned mutable state present after the host-operator epic, including
   authentication and host proposal/schedule state, shares the selected
   transactional and recovery boundary; the root-owned privileged audit remains
   an explicit external boundary
   (test: `test_post_host_state_uses_declared_persistence_boundary`).
5. manual: with two agents completing and a host proposal changing state at the
   same time, their sessions, outcomes, and proposal remain visible after
   restarting Scufris.

## Child Tasks

Four lanes, each run in dependency order. Derive live status with
`tatr frontier 20260729-102145`.

Lane A - persistence decision:

- [ ] 20260729-102146 (p85, v0.2.0) inventory app-owned mutable state and
      reproduce the write races
- [ ] 20260801-100405 (p84, v0.2.0) choose the persistence mechanism,
      migration, and recovery policy

Lane B - persistence implementation:

- [ ] 20260729-102147 (p80, v0.2.0) introduce the transactional persistence
      core and migrate the project store
- [ ] 20260801-100409 (p79, v0.2.0) migrate agent, session, outcome, settings,
      and reasoning state
- [ ] 20260801-100413 (p78, v0.2.0) migrate auth, host, schedule, and digest
      state with a legacy JSON import path

Lane C - backend truth:

- [ ] 20260729-102148 (p75, v0.2.0) extract the backend-aware orchestrator
      diagnostics service
- [ ] 20260801-100415 (p74, v0.2.0) delegate legacy `/api/agent/*` routes to
      orchestrator diagnostics
- [ ] 20260801-100419 (p73, v0.2.0) align Telegram and the UI with
      orchestrator diagnostics

Lane D - assembly refactor:

- [ ] 20260801-100425 (p72, v0.2.0) characterize app routes and extract the
      auth and host routers
- [ ] 20260801-100441 (p71, v0.2.0) extract the orchestrator-turn and
      agent-run services
- [ ] 20260729-103712 (p70, v0.2.0) extract the remaining routers and reduce
      `create_app` to assembly

## Decisions

- Pending 20260801-100405 SPIKE.md and DECISION.md: persistence mechanism,
  migration boundary, recovery policy, and support for ordered append-only
  events plus atomic state-and-event commits.
- Recorded 20260729-102146 SPIKE.md: the state inventory, mutator matrix, and
  the reproduced write race the decision argues against. Two findings bear on
  Done Means 4: host proposals have NO persisted store today
  (`host_actions.HostActionStore` is memory-only, rebuilt from the root helper),
  and auth sessions are already lock-protected. 20260801-100405 owes an explicit
  answer on whether the host proposal store joins the boundary.

## Manual Acceptance

- (pending) 20260801-100405: the durability and migration tradeoffs of the
  selected persistence architecture.
- (pending) 20260801-100413: existing local state migrates without losing
  projects, agents, sessions, outcomes, settings, authentication state, or
  app-owned host state.
- (pending) 20260801-100419: backend/account information feels consistent
  across the landing page, agent settings, and Telegram.

## Downstream v0.2.0 Readiness

This epic is the persistence/service foundation for the actor-aware
orchestrator direction. It does not implement that product behavior. The
v0.2.0 readiness work scheduled after it is:

- 20260729-220835: conversation/flow architecture spike plus interactive HTML
  mockup.
- 20260729-102151 through 20260729-102154: deterministic browser QA.
- 20260729-102158: structured tatr task metadata.
- 20260729-102203: durable agent-run activity and hierarchy.
- 20260729-102205 and 20260729-102206: reusable agent preset architecture and
  implementation.
- 20260729-102202: responsive/accessibility baseline.
