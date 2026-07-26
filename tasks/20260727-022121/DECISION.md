# Decision: sub-agent work steering is backend-agnostic, not flow-skill-dependent

- TASK: 20260727-022121
- STATUS: ACCEPTED
- DATE: 2026-07-27

## Context

The goal is that a spawned sub-agent, once launched to implement a task, knows
to actually do the work and to signal blockers via `request_input`. The user's
phrasing was "the spawned agent should know that it needs to use flow and
request_input".

## The fork (mutually exclusive constraint)

`flow` is a Claude Code skill living in `~/.claude/skills/flow`. A sub-agent on
the **claude** backend can load it; a sub-agent on the **codex** backend
cannot - codex has no Claude Code skill mechanism. The reported failure is
exactly this: a codex agent told (via its goal) to "use the flow skill"
produced framing text and stopped with 0 tool calls, because there was no such
skill to run and no actionable steps on the turn prompt.

So "steer the sub-agent to use the flow skill" and "one steering path that
works on both backends" cannot both hold: the flow skill is claude-only.

## Options

- A. Depend on the flow skill: steer sub-agents to run `/flow`. Clean for
  claude; for codex the goal prompt must carry flow-like instructions anyway,
  so codex effectively gets a second, divergent path - fragile, and it is the
  shape that already failed.
- B. Backend-agnostic work clause (CHOSEN): `AGENT_STEERING_PREAMBLE` gives
  every sub-agent actionable turn-prompt steps (implement the assigned task
  end-to-end, run the project's checks, do not stop at a plan, call
  `request_input` when blocked). It MAY additionally mention "use the flow
  skill if available" for claude, but the instruction stands without it, so
  codex behaves correctly too. One path, no hard skill dependency.

## Decision

Chose B. Confirmed with the operator at the plan gate (2026-07-27). This also
tracks the ledger lesson `codex-tool-choice-only-steers-via-the-turn-prompt`:
the instruction has to ride the turn prompt as concrete steps, not lean on a
soft channel (a skill file) codex never reads.

## Consequence

Scope this cycle is steering only. The steering fix plausibly addresses the
root cause of the "did nothing" run, but if a live delegated run still stalls,
that execution bug is a separate follow-up task, not part of this one.
