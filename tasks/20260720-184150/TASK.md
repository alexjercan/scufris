# Spike: multiple agents + workflows for scufris

- PRIORITY: 25
- TAGS: spike, agent
- KIND: SPIKE
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Spike (do NOT build): what would "multiple agents with workflows" mean for
scufris, and is it worth building? The user asked to "think about adding
multiple agents somehow with workflows". This is the fuzziest part of the
settings-console goal (umbrella 20260720-183719) and was explicitly scoped as
spike-only. Output: a SPIKE.md with a recommended direction and seeded
direction-level tatr tasks - no implementation.

## What the spike must decide

- What is a "second agent" here? A different codex model/backend/tool-set
  (a profile - which task 3 already delivers), a genuinely separate agent
  process, or an orchestration of agents that hand off?
- What is a "workflow"? A saved multi-step prompt/tool sequence, a
  multi-agent pipeline, or something like codex's own skills
  (`~/.codex/skills`, noted empty in tasks/20260720-122301/SPIKE.md)?
- How does it relate to what already exists: config profiles (task 3),
  sessions, the projects concept (tasks/20260720-182842), and codex's
  single-session-per-turn lock (`app.py` serializes turns)?
- Honest cost/benefit for a single-operator homelab tool: is this worth
  building, deferring, or dropping?

## Steps

- [x] Ground in the code + codex: re-read `scufris/agent.py` (backends, the
      turn lock), config profiles (task 3), the projects spike, and probe
      codex skills / any multi-agent primitive (`~/.codex/skills`,
      `codex app-server` thread model).
- [x] Diverge: enumerate candidate concepts (profiles-as-agents; separate
      agent registry; workflow = saved prompt/tool macro; workflow = codex
      skills; full orchestration) with pros/cons/unknowns.
- [x] Converge on a recommendation (may legitimately be "defer" or "drop"),
      concrete enough for /plan to expand.
- [x] Write `tasks/<this-id>/SPIKE.md` (spike format) and seed direction-level
      tasks if the recommendation is to build; close this spike task.

## Definition of Done

- A `SPIKE.md` exists with Question, Options considered, a Recommendation, and
  Next steps (cmd: `test -f tasks/<this-id>/SPIKE.md`).
- Any seeded tasks are created and lint clean, or the doc explicitly concludes
  "do not build" (cmd: `tatr check --ledger LESSONS.md`).
- manual: the user agrees the recommended direction (or the decision to defer)
  matches their intent for "multiple agents with workflows".

## Notes

- This is a `/spike`, not a `/work` task: research + direction, no shipped code.
- Relevant priors: tasks/20260720-122301/SPIKE.md (codex skills empty,
  no /commands in exec/app-server), tasks/20260720-182842/SPIKE.md (projects),
  config profiles (task 3 of this goal).
