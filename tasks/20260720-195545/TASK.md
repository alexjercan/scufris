# Workflows via codex skills: surface + manage the user skills dir in the console

- STATUS: OPEN
- PRIORITY: 12
- TAGS: feature,agent,spike

## Goal

Adopt codex SKILLS as the "workflow" primitive instead of building an
orchestration engine. codex already discovers and runs SKILL.md skills from
`~/.codex/skills/` (system skills ship there; a user dir is available). Surface
the available skills in the operator console and let the operator manage the
USER skills dir (list first; add/remove later). A "workflow" is thus a
reusable, codex-run recipe an agent can be steered to - not a scufris pipeline.

## Notes

- Spike: tasks/20260720-184150/SPIKE.md (Option C; orchestration/Option D was
  DROPPED - do not build a hand-off engine). Read it first.
- Stepless: run `/plan` when picked up.
- Open question from the spike: read-only skill listing first (safe) vs full
  add/remove of user skills. Lean read-only first.
- Composes with the agents task (20260720-195543): an agent's persona can point
  at a skill.

