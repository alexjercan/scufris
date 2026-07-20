# Agents: named personas on top of profiles (system prompt + model + tools)

- STATUS: CLOSED
- PRIORITY: 15
- TAGS: feature, agent, spike

## Goal

Turn the config-profile concept into a named AGENT: a profile PLUS a
`system_prompt` persona injected via the existing steering preamble
(`agent.py:_steer`, the only channel codex reliably obeys). An agent =
`{name, system_prompt, model, backend, enabled_tools}`. A session selects an
agent, so the operator can have e.g. a "sysadmin" agent and a "journal" agent
with different instructions and tools. Reuse the profile store (an agent IS a
profile + a persona); do NOT build a separate registry unless planning shows
it is needed.

## Notes

- Spike: tasks/20260720-184150/SPIKE.md (Option B). Read it first.
- Stepless: run `/plan` when picked up.
- Open question from the spike: per-session agent selection needs the
  sessions/projects scoping (tasks/20260720-182842) to settle first - decide
  session-scoping before building selection.
- Depends conceptually on the profile store (tasks/20260720-184138, landed).
- SUPERSEDED (user feedback 20260720): "multiple agents" is not personas on one
  codex - it is PER-PROJECT agents (scufris as a project orchestrator). See
  tasks/20260720-184150/SPIKE.md "Revision 1". The persona idea, if anything,
  becomes a per-project agent's config under that concept (phase P1/P2). Closing
  this task as the wrong axis; work continues under the projects concept.

