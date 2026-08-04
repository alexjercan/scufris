# Per-project pinned context via the steering preamble

- PRIORITY: 14
- TAGS: feature, agent, spike
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Let each project carry a pinned context note (`context_md` in the project
record) that is injected into every turn run in that project, riding the
EXISTING steering-preamble channel (the proven instruction path;
AGENTS.md-via-`-C` is unreliable per the `codex-tool-choice-only-steers-via-
the-turn-prompt` lesson). Editing the note is a small UI + endpoint. Keep it
sandbox-safe (preamble, not a file written into the project dir).

## Notes

- Spike: tasks/20260720-182842/SPIKE.md (open question: preamble vs AGENTS.md
  channel - defaulting to preamble; an AGENTS.md-read probe is a later option).
- Depends on the backend task (20260720-182938) for the project store.
- Stepless: run `/plan` when picked up.


- SUPERSEDED (20260720) by projects orchestrator P0 (umbrella 20260720-210347): the minimal cwd-grouping design is re-cut into the first-class Project entity. The session<->cwd scoping and per-turn -C parts here defer to P1/P2 (per-project agent). See tasks/20260720-184150/SPIKE.md Revision 1.
