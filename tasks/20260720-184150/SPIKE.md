# Spike: multiple agents + workflows for scufris

- DATE: 20260720-184150
- STATUS: RECOMMENDED
- TAGS: spike, agent

## Question

The user asked to "think about adding multiple agents somehow with workflows".
What would that concretely mean for scufris, and is it worth building? A good
answer names what a "second agent" and a "workflow" ARE here, relates them to
what already exists (config profiles, sessions, projects, the turn lock), and
gives an honest build/defer/drop verdict with seeded direction-level tasks.
This was scoped spike-only (umbrella 20260720-183719); NO implementation.

## Context (grounded on the host, codex present)

- **All codex turns are serialized.** `scufris/app.py` holds a single
  `chat_lock` (`asyncio.Lock`) around every turn ("Codex sessions are not
  concurrency-safe; serialize chat turns"). So genuinely CONCURRENT agents are
  not available without lifting that lock, which codex's own non-concurrency
  forbids. Multi-agent means SEQUENTIAL at best.
- **Config profiles already exist (task 20260720-184138).** A profile is a
  named, runtime-switchable override set: `{agent_model, agent_backend,
  agent_tools_enabled, disabled_tools, mcp_servers, ...}`. So "a different
  model / backend / tool-set you can switch to" is ALREADY shipped - that is
  most of what "another agent" informally means.
- **codex has a native skills mechanism, now populated.** `~/.codex/skills/`
  exists with `.system/` skills (`imagegen`, `openai-docs`, `plugin-creator`,
  `skill-creator`, `skill-installer`); codex's base instructions have a "Using
  skills" section (SKILL.md discovery). A USER skills dir is available. (The
  earlier spike tasks/20260720-122301 saw it empty; it is now seeded by codex.)
  This is codex's built-in "reusable multi-step recipe" primitive.
- **No custom slash-commands in exec/app-server** (tasks/20260720-122301):
  `/prompts` is a TUI-only feature; a web command palette would be ours.
- **Steering rides the turn prompt.** `agent.py:_steer` prepends a preamble to
  the prompt (the only channel codex reliably obeys, per
  `codex-tool-choice-only-steers-via-the-turn-prompt`). A per-agent PERSONA
  (system prompt) would ride the same proven channel.
- **Projects (tasks/20260720-182842) = a cwd.** A per-project default agent is
  a natural later join, but out of scope here.

## Options considered

- **A - Profiles ARE agents (reframe only).** Relabel profiles as "agents" and
  add per-session selection. Pros: near-zero build, rides task 3. Cons: a
  profile has no persona/instructions, so it is a weak notion of "agent" - two
  profiles with the same model behave identically.

- **B - Named agent personas on top of profiles.** An "agent" = `{name,
  system_prompt, model, backend, enabled_tools}` - a profile PLUS a persona
  (the system prompt injected via the steering preamble). A session picks an
  agent. Pros: a real, distinguishable "agent" (a "sysadmin" vs a "journal"
  agent with different instructions + tools); modest extension of the existing
  profile store + steering channel; honest to the user's ask. Cons: needs a
  persona field, per-session agent selection, and UI.

- **C - Workflows = codex skills (native).** A "workflow" = a saved SKILL.md
  the agent can discover and run. scufris manages the user `~/.codex/skills/`
  dir (list/add/remove) and surfaces skills in the UI; codex does the actual
  discovery/execution. Pros: leans on a real codex mechanism instead of a
  bespoke engine; cheap; composes with agents (an agent can be steered to a
  skill). Cons: skills are codex-run (we observe, not orchestrate); value
  depends on the user actually writing skills.

- **D - Multi-agent orchestration (hand-off pipeline).** Agent A does step 1,
  hands to agent B, etc. Pros: the maximal reading of "workflows". Cons: the
  turn lock + non-concurrency-safe sessions make it sequential-and-serialized
  and expensive; a real orchestration engine (state, hand-off, error handling)
  is a large build with low payoff for a single-operator homelab. High risk of
  building a framework nobody drives.

- **E - Do nothing / defer.** Profiles already cover the practical "switch
  model/tools" need. Cost: leaves the user's "agents + workflows" ask
  unanswered beyond profiles.

## Recommendation

**Build B (agents = personas on profiles) and C (workflows = codex skills);
DROP D (orchestration); this supersedes A and E.**

- **"Multiple agents" -> B.** Extend the profile concept into a named AGENT
  with a `system_prompt` (persona) injected through the existing steering
  preamble, plus its model/backend/tool-set (reuse the profile store). A
  session selects an agent. This gives genuinely different agents (persona +
  tools), not just model toggles, at a modest cost because the store and the
  steering channel already exist. Superset of A.
- **"Workflows" -> C.** Adopt codex SKILLS as the workflow primitive rather
  than building an orchestration engine: let the operator manage the user
  skills dir and see available skills; an agent's persona can point at a skill.
  A "workflow" is thus a reusable, codex-run recipe, not a scufris pipeline.
- **DROP D.** True multi-agent hand-off is blocked by the turn lock and codex
  session non-concurrency, and its value for one operator does not justify an
  orchestration framework. Revisit only if a concrete recurring hand-off need
  appears - note it here if it does.

Why this over the alternatives: it answers the user's ask with real,
distinguishable agents and a real workflow mechanism while reusing the profile
store and the steering channel (small build), and it explicitly refuses the one
expensive, low-value path (D) instead of drifting into it.

## Open questions

- Does an agent's persona belong on the profile record (`agents.json` extends
  the profile store) or a separate registry? Lean: extend the profile store -
  an agent IS a profile + a persona.
- Per-session agent selection: store the chosen agent id on the session (codex
  cwd/metadata) or a scufris side-table? Needs the projects/sessions design
  (tasks/20260720-182842) to settle first - so B should land AFTER or with a
  clear session-scoping decision.
- Skills UX: read-only listing first, or full add/remove of user skills? Lean
  read-only listing first (safe), editing later.

## Next steps

Direction-level tasks this spike seeded (for a FUTURE flow, not this goal's
umbrella; `/plan` breaks them into steps):

- tatr 20260720-195543: Agents = named personas on top of profiles
  (system_prompt + model + tools), per-session selection.
- tatr 20260720-195545: Workflows via codex skills - surface + manage the user
  skills dir in the operator console.

DROPPED (documented, no task): multi-agent orchestration / hand-off pipelines.

## Fix record

(Appended by each implementing task as it lands.)
