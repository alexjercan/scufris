# Per-project skills + custom tools (surface + manage), scufris orchestrator P3

- PRIORITY: 12
- TAGS: feature, agent, spike, projects
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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


## Resolution: WONT-DO (closed 20260723)

The READ-ONLY surface half of this task shipped under umbrella
20260723-225437 (tasks 20260723-225616 backend + 20260723-225621 frontend):
a project's per-project skills (.claude/skills, codex .codex/skills) and custom
tools / MCP servers (.mcp.json + .claude/settings*.json; codex .codex/config.toml)
are discovered provider-aware and surfaced read-only on the agent settings page
(GET /api/agents/{id}/capabilities). This satisfied the task's "read-only
listing first" open question.

The remaining MANAGE half (letting the operator add/remove/edit skills or MCP
servers from the UI) is intentionally DECLINED: we do not want the UI to write
skills or MCP-server config into a project tree. Managing those stays a
file-level / editor concern, out of scope for scufris. Closing won't-do rather
than re-scoping, since the only remaining work was the management surface we are
choosing not to build.
