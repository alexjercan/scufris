# Spike: define the actor-aware orchestrator conversation and flow-control model

- STATUS: OPEN
- PRIORITY: 69
- TAGS: spike, v0.2.0, agents, orchestrator, projects, telegram, frontend

## Story

As the Scufris operator, I want an accepted architecture and visual interaction
model for an actor-aware orchestrator dashboard, so that the future feature
separates its durable conversation from provider sessions, technical activity,
workflow truth, and enforcement audit before production schemas or UI are
built.

The spike must make the end-to-end product concrete: `/` is a Scufris-owned
conversation shared with Telegram; Projects joins tatr tasks, lifecycle gates,
agent assignments, runs, worktrees, reviews, and artifacts; Agents remains the
direct native-session console; the orchestrator coordinates stage-specific
specialists and host capabilities.

## Steps

- [ ] Ground the spike in the post-host/post-durability architecture and the
      existing session registry, outcome/wake loop, provider transcript readers,
      Telegram transport, Projects page, agent records, supervisor, and open
      v0.2.0 prerequisite tasks.
- [ ] Compare at least three conversation ownership approaches: provider
      session as product conversation, Scufris-owned full provider transcript,
      and the recommended semantic Scufris conversation above provider-owned
      native sessions. Include "do nothing" and record the fidelity, caching,
      attribution, recovery, and backend-portability tradeoffs.
- [ ] Define the logical records and invariants for conversations, actors,
      semantic conversation events, channel bindings/delivery, provider-session
      bindings, agent runs, activity events, workflow assignments, approvals,
      artifacts, and correlation/idempotency keys. Keep the model compatible
      with the persistence decision from 20260729-102146.
- [ ] Define the event taxonomy and projections: what appears in the human
      conversation, what stays in technical activity, what is authoritative
      enforcement audit, what remains only in a provider transcript, and how
      untrusted agent reports are distinguished from operator instructions.
- [ ] Define orchestrator context assembly for a logically stateless mediator:
      system/project policy, versioned summary, recent semantic events, pending
      decisions/workflows, available presets/capabilities, compaction, and the
      role of an optional provider session as a cache rather than truth.
- [ ] Define the project flow-control state machine and server-side launch
      guards for task creation/planning, plan approval, work, review,
      changes-requested loops, compound, and land. Tatr files remain
      authoritative and are re-read through 20260729-102158 before launch.
- [ ] Define reusable-preset, agent-instance, run, task/stage assignment,
      worktree/branch/commit, parent conversation/run, and native provider
      session relationships, including reusable versus ephemeral specialist
      agents and system/hidden-agent visibility.
- [ ] Define web/Telegram semantics: one source of conversation truth,
      conversation selection/new-chat behavior, ordered replay, delivery
      deduplication, edits/batching, notification noise policy, approvals, and
      reconnect/restart recovery.
- [ ] Build an interactive static HTML mockup at
      `tasks/20260729-220835/mockup.html` with fixture data and no production
      integration. Show linked `/` and Project views through the scenario:
      no task -> planning agent -> plan approval -> implementation WIP ->
      report -> review -> changes requested or approved -> compound/land.
      Include agent attribution, task WIP/agent indicators, branch/artifact
      links, unavailable-action explanations, pending approvals, and the same
      semantic conversation represented in a Telegram-sized view.
- [ ] Play through the mockup at desktop and phone widths, iterate with the
      user until the information hierarchy and lifecycle controls are accepted,
      and record remaining product questions rather than hiding them in the
      implementation plan.
- [ ] Write `SPIKE.md`, record accepted load-bearing choices in `DECISION.md`,
      and explicitly supersede the old "drop orchestration pipelines" decision
      if the project flow coordinator remains the recommendation.
- [ ] Only after the architecture and mockup are accepted, seed the backlog
      epic `EPIC: Make Scufris an actor-aware orchestrator dashboard` and its
      direction-level tasks. Reuse existing task-board, artifact, activity, and
      orchestrator-preset tasks rather than duplicating them.

## Definition of Done

- The spike compares the alternatives and decides conversation ownership,
  provider-session use, actor attribution, context projection, channel
  delivery, workflow authority, and recovery
  (cmd: `rg -n "provider session|conversation|actor|context|Telegram|workflow|recovery" tasks/20260729-220835/SPIKE.md`).
- The accepted architecture records semantic conversation, technical activity,
  enforcement audit, and provider transcript as distinct sources/projections,
  with stable correlation and idempotency invariants
  (cmd: `test -f tasks/20260729-220835/DECISION.md && rg -n "semantic|activity|audit|provider|correlation|idempoten" tasks/20260729-220835/DECISION.md`).
- The static mockup exists in the task folder and demonstrates the complete
  plan/work/review loop across the main conversation, Project workspace, and
  Telegram-sized projection
  (cmd: `test -f tasks/20260729-220835/mockup.html && rg -n "planning|implementation|review|Telegram" tasks/20260729-220835/mockup.html`).
- The future epic and seeded tasks are created only after acceptance and cite
  this spike instead of restating its decisions
  (cmd: `set -o pipefail; rg -n "Spike: tasks/20260729-220835/SPIKE.md" tasks/*/TASK.md | rg -v "^tasks/20260729-220835/TASK.md:"`).
- manual: the user can play through the mockup and understands who said what,
  what is running, what decision is pending, which action is legal next, and
  where to inspect the native agent transcript or technical activity.

## Notes

- Depends on: 20260729-124655 and 20260729-102145.
- This is research and an HTML validation artifact, not production code.
- Prior research/decisions to reconcile:
  `tasks/20260720-184150/SPIKE.md`,
  `tasks/20260720-221748/SPIKE.md`,
  `tasks/20260723-001256/SPIKE.md`,
  `tasks/20260724-111839/SPIKE.md`,
  `tasks/20260724-132713/DECISION.md`, and
  `tasks/20260727-022121/DECISION.md`.
- The mockup belongs in this task folder as agreed; do not place it in `docs/`
  or wire it into the shipped webpack application.
- The full provider transcript remains native unless the spike overturns that
  with evidence. The proposed Scufris conversation is a semantic mediator
  history above it, not a lossy transcript replacement.
- No implementation epic is created during this planning update. Its exact
  children depend on the accepted spike and mockup.

## Flow State

- FLOW STEP: PLANNING
