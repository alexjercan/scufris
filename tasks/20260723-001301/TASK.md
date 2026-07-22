# Harden the orchestrator<->sub-agent session view on the agent page

- STATUS: OPEN
- PRIORITY: 30
- TAGS: bug,agents,frontend,ui

## Story

As the operator, I want the orchestrator<->sub-agent conversation to render reliably
on the agent's page (live, and on reload / reselect), because today it works but is
flaky.

## Context

Orchestrator turns against a sub-agent (`message_agent` / `run_agent`) already run on
the shared supervisor + event bus and land on the agent page. "Sometimes not well"
is reattach/replay robustness - the same class as several open UI items, not a new
mechanism.

## Direction

- [ ] Harden SSE reattach-on-select and transcript replay for the per-agent session
      view: a reselect/reload must rebuild the full transcript AND continue an
      in-flight turn.
- [ ] Persist/replay tool-call chips + per-turn usage across reload for agent
      sessions (the orchestrator chat already does this - reuse it).
- [ ] Do the persisted session registry task FIRST - stable session keys remove a
      chunk of the flakiness underneath this.

## Definition of Done

- Reselecting or reloading an agent mid-turn shows the full transcript and continues
  streaming. (manual + test where feasible)

## Notes

- Overlaps existing backlog - coordinate / possibly merge rather than duplicate:
  `20260721-112428` (F0 SSE reattach on select), `20260720-020356` (stream tokens
  end-to-end), `20260720-122513` (persist tool-call chips across reload).
- Depends on the `(agent_id -> session_id)` persisted registry task.
