# Per-project skills + custom tools (surface + manage), scufris orchestrator P3

- STATUS: OPEN
- PRIORITY: 12
- TAGS: feature,agent,spike,projects

## Goal

Adopt codex SKILLS as the "workflow" primitive instead of building an
orchestration engine. codex already discovers and runs SKILL.md skills from
`~/.codex/skills/` (system skills ship there; a user dir is available). Surface
the available skills in the operator console and let the operator manage the
USER skills dir (list first; add/remove later). A "workflow" is thus a
reusable, codex-run recipe an agent can be steered to - not a scufris pipeline.

## Notes

- Spike: tasks/20260720-184150/SPIKE.md, Revision 1, phase P3. Read it first.
- RE-SCOPED (user feedback 20260720): from "global codex skills in the console"
  to PER-PROJECT skills + custom tools, surfaced on a project's page - part of
  the projects-orchestrator concept. codex skills (`~/.codex/skills`, now
  seeded with `.system/*`) remain the run mechanism, but scoped per project
  (project-local skill dirs + per-project MCP servers).
- Depends on the projects P0/P1 work (a first-class Project entity + page)
  landing first - do not build before projects exist.
- Stepless: run `/plan` when picked up.
- Open question: read-only skill listing first (safe) vs full add/remove.
  Lean read-only first.

