# Spike: multiple agents + workflows for scufris

- DATE: 20260720-184150
- STATUS: SUPERSEDED by Revision 1 (see bottom) - the user reframed "multiple
  agents" as PER-PROJECT agents (scufris as a project orchestrator), which
  changes the recommendation. The original analysis below is kept as history.
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

## Revision 1 (user feedback, 2026-07-20): scufris as a project orchestrator

The original recommendation (personas-on-profiles + codex skills) was too small
and aimed at the wrong axis. The user's actual intent, in their words:

> "we have projects, scufris is an orchestrator ... a Project page, each one
> with its own tatr tasks, and config, and details, language, description etc.
> and then each project has its own agent configured (or multiple agents for
> multiple things) - e.g. we can have `claude code` setup for a project to
> 'implement and do things for that project' and we connect from the frontend
> to that project to do things via the agent we configured ... like a frontend
> for my sesh multiplexer + convenient tools to work with the project from the
> UI ... used mainly for spec-driven development ... skills of that project,
> custom tools for that project ... a Dev environment that lets us 'vibe code'
> with tools + skills + tatr + other tools."

Also stated: profile SWITCHING is not wanted (that role is filled by `login`),
and configuring the server via API calls is preferred over per-setting UI toggles.

### The reframed vision

**A "project" is the organizing unit, and scufris is its orchestrator.** A
project is a workspace (a directory) that carries:

- metadata: name, language, description, cwd;
- its own tatr tasks (the SPECS - this is spec-driven development);
- its own configured agent(s) - possibly DIFFERENT backends for different jobs
  (codex, Claude Code, ...), e.g. "the implementer agent for project X";
- its own skills and custom tools.

You open a project in the web UI and drive its agent(s) to work its specs. It
is a web frontend over the same things `sesh` fronts in the terminal (project
dirs) plus tatr (specs) plus agents plus skills/tools - a single-operator dev
environment for spec-driven "vibe coding". "Multiple agents" therefore means
PER-PROJECT agents, NOT personas-on-one-codex (Option B, dropped) and NOT an
orchestration pipeline (Option D, still dropped).

### How it maps onto what already exists

- **Project dir** <- codex's cwd tagging + `sesh`'s `~/personal`/`~/work`
  convention (the earlier projects spike, tasks/20260720-182842).
- **Per-project tatr** <- `tatr` is already directory-scoped (`tatr -r <root>`),
  and scufris already wraps it as MCP tools (`tatr_ls/show/new`).
- **Per-project agent** <- the `Agent` protocol already abstracts
  chat/stream/session; today only codex implements it.
- **Per-project skills/tools** <- codex skills (`~/.codex/skills`) + per-project
  MCP servers (the `mcp_servers` config, scoped to a project).
- **The spec-driven loop** <- literally scufris's own AGENTS.md
  plan-work-review flow, surfaced in the browser.

The pieces exist; the vision is mostly about ELEVATING the project to a
first-class entity and generalizing the agent layer.

### This flips the earlier projects spike

tasks/20260720-182842 deliberately chose the MINIMAL project (Option A: cwd +
`{name, context_md}`, rejecting the first-class "Option B" object). Under this
vision that choice inverts: a project genuinely needs a first-class record
(name, language, description, agent config(s), skills, tools) that a cwd cannot
carry - so **Option B is now the right call**, and the projects spike's seeded
tasks (182938/182953/182959) should be re-scoped upward into this concept.

## Review of the vision (asked for)

An honest critique - what is strong, what is risky, and a phased path.

### Strengths

- **Coherent with the grain of the codebase.** cwd-tagging, dir-scoped tatr, the
  Agent protocol, the steering channel, the writable config store (this goal)
  and the-den/sesh conventions all already point at "a project is a dir with
  its own agent + tools". This is elevation, not invention.
- **A real differentiator.** A structured, spec-driven, per-project agent
  console is meaningfully different from a raw terminal or a generic chat - it
  is the plan-work-review loop the user already runs, made visual and
  multi-project.
- **Single-operator scope keeps it tractable.** No multi-tenant auth, no
  sharing - the hard parts of an "agent IDE" are off the table.

### Risks and the hard parts

1. **Multi-backend is the biggest lift and the main risk.** "Claude Code set up
   for a project" means driving agent CLIs beyond codex. The `Agent` protocol
   abstracts the surface, but the concrete backends (codex exec / app-server)
   are hardcoded and each real backend has its OWN streaming, session and MCP
   model (Claude Code: `claude -p --output-format stream-json`, its own
   permission/MCP config). Recommendation: force the abstraction honest by
   proving exactly TWO backends (codex + Claude Code), not N speculative ones,
   and PROBE each on the host before committing (the `probe-runtime-on-target-
   host-early` lesson).
2. **Concurrency needs a per-project lock.** Today a single global `chat_lock`
   serializes ALL turns. A multi-project orchestrator wants project A's agent
   to run while you look at project B. Different agent processes in different
   cwds are independent, so the lock must become PER-PROJECT (per agent
   instance) - tractable, but a real change from the current single bottleneck.
3. **Write access is a genuine escalation.** "Implement and do things" means the
   agent WRITES (not today's read-only codex sandbox). Each backend's
   sandbox/approval model differs (codex `--sandbox`, Claude Code permissions).
   Per-project write-enablement must be a deliberate, gated decision, not a
   default - this is the sharpest safety edge of the whole idea.
4. **Scope explosion.** Per-project metadata + agents + skills + tools + tatr +
   a dispatch/watch loop is, in sum, a lightweight agent IDE. It only ships if
   phased; built all at once it stalls.
5. **"Spec-driven" vs "vibe code" tension.** Lean on the STRUCTURE (tatr tasks
   as specs, the review gate) as the spine - that is the differentiator - with
   freeform chat as an escape hatch, not the center. If it becomes just chat in
   a project dir, it is a worse terminal.
6. **Don't reinvent sesh/tmux.** scufris is the WEB view over the same project
   dirs, not a tmux replacement (a browser cannot attach to a tty). Keep that
   boundary explicit or the two tools fight.

### Recommended phased path (each phase its own future flow)

- **P0 - Project as a first-class entity.** The rich store
  (`projects.json`: `{cwd, name, language, description}`), a Project PAGE (not
  just a sidebar switcher), and its per-project tatr view. Supersedes/absorbs
  the minimal projects tasks (182938/182953/182959).
- **P1 - Per-project agent CONFIG.** Which backend/model/tools an agent uses,
  stored per project - still codex-only, but configured per project (reuse the
  writable-config store from this goal; expose it as the API-first config the
  user asked for).
- **P2 - Multi-backend + per-project lock.** Add Claude Code as the second
  backend to force the `Agent` abstraction honest; per-project backend
  selection; the per-project turn lock for concurrency; the write-access gate.
- **P3 - Per-project skills + custom tools.** Project-local skill dirs + per-
  project MCP servers, surfaced on the Project page.
- **P4 - The dispatch loop.** Dispatch an agent to work a tatr task, watch it
  stream, review the result - a web frontend for plan-work-review.

### Verdict

Build it, as the PROJECTS concept expanded - the multi-agent question resolves
to per-project agents. Start at P0/P1 (cheap, rides this goal's config store and
the projects spike) and treat P2's multi-backend + write-access as the real
research/risk to probe before over-building. Drop personas-on-profiles and
orchestration pipelines. Reframe (do not delete) the projects spike around
Option B.

### Consequences for existing work (recorded)

- **Config profiles (this goal's tasks 184138/184149) are not the wanted axis.**
  The user switches config via `login`, not a profile switcher; per-project
  agent config (P1) is the real config-switching story. The profile switcher UI
  and the profiles backend are candidates for removal or repurposing - flagged
  for a user decision, not auto-removed.
- **Config-via-API is already delivered** by task 184136 (`PATCH
  /api/agent/config` + the store); P1 extends it to per-project scope. The UI
  toggles are a thin layer over that API and can stay or be de-emphasized.
- **Seeded tasks reconciled:** `20260720-195543` (personas-on-profiles) is
  CLOSED as superseded; `20260720-195545` (codex skills) is re-scoped to
  "per-project skills/tools" under P3.

## Next steps (Revision 1)

Direction-level work for FUTURE flows, in phase order (each needs `/plan`, and
P0 should re-cut the earlier projects spike's tasks rather than run beside them):

- P0/P1: expand tasks/20260720-182842 (projects) to Option B - a first-class
  Project entity with a page, metadata and per-project agent config.
- P2: spike + build multi-backend agents (codex + Claude Code), per-project
  lock, write-access gate.
- P3: `20260720-195545` re-scoped to per-project skills/tools.
- P4: spec-driven dispatch/watch/review loop.

(No new tatr tasks created here beyond the reconciliation above; P0-P4 are
seeded as direction and belong to future `/plan` runs so they are not built
speculatively.)

## Fix record

(Appended by each implementing task as it lands.)
