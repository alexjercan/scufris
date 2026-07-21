# Spike: agent-surface unification - routing, shared settings page, orchestrator as a hidden default

- DATE: 20260721
- STATUS: RECOMMENDED
- TAGS: spike, agents, frontend, backend
- UMBRELLA: 20260721-234126 (Agents UX v3)

## Question

How do we collapse the landing page and the per-agent pages into ONE agent
surface - same chat + settings UI for every agent, the orchestrator merely a
"hidden default" agent exposed at `/` (and `/settings`) instead of under
`/agents/...` - without a rewrite? Specifically: (a) the routing/component-sharing
model; (b) how the orchestrator stays a clean hidden, projectless special case;
(c) how to unify the settings surface, which today is SPLIT (the orchestrator's
settings are the `/settings` PAGE; a project agent's settings are a MODAL on
`/agents/<id>`); (d) the richer settings (backend->auto model, model autocomplete,
permission mode, detailed context/memory/account panels, orchestrator's
multi-session extra).

## Context (what already exists)

- **Chat is ALREADY converged** (B5d, 20260721-180222): `createAgentChat(root,
  config)` in `web/src/agent-chat-view.ts` is the ONE chat component, mounted on
  `#agent-chat` by both the landing entry (`agent-view.ts` -> orchestrator) and
  the detail entry (`startAgentChat` -> `agentIdFromPath`). So requirement (4)'s
  CHAT half is done; this EPIC is mostly the SETTINGS half + routing symmetry +
  hiding the orchestrator.
- **Routing** is webpack multi-page + a backend SPA fallback: entries `agent`
  (`/`, index.html), `settings` (`/settings/`), `agents` (`/agents/`),
  `agent-detail` (served for `/agents/<id>` and `/agents/<id>/<rest>` via
  `_agent_detail_shell`). The catch-all `@app.get("/agents/{agent_id}/{rest:path}")`
  ALREADY serves the detail shell for `/agents/<id>/settings` - so a settings
  SUB-page needs no new backend route, just an entry/mount that reads the sub-path.
- **Two parallel API worlds.** Singular `/api/agent/*` (the orchestrator):
  `config`, `context`, `usage`, `memory`, `account`, `health`, `info`, `tools`,
  `sessions` + `session`/`session/fork` (multi-session), `profiles`,
  `mcp_servers`. Plural `/api/agents/{id}/*` (any agent, incl. orchestrator):
  `chat`, `transcript`, `status`, `fork` (revert), `run`, `events`; plus CRUD
  `/api/agents/{id}` and `/api/agents/backends`.
- **Data sources**: `read_context(codex_home, session_id)` is PER-SESSION (so a
  per-agent context is just its own session's); `read_usage`/`read_memory`/
  account are ACCOUNT-level (per codex/claude home), i.e. the account backing the
  agent, not per-agent-session. `/status` already carries per-agent context.
- **The orchestrator** is a synthetic record in `AgentStore` (B5a): undeletable,
  `project_id=""` (ALREADY projectless, runs in server cwd), backend/model from
  settings (`canonical_backend(settings.agent_backend)` + `default_model_for`).
  It appears FIRST in `/api/agents` (so it shows in the `/agents` list today) and
  409s on the per-agent PATCH (its config belongs to the settings store).
- **Settings split**: `/settings` = `settings-view.ts` reading the GLOBAL
  `/api/agent/config` (agent_enabled, backend, model, auth, tools, mcp_servers,
  profiles) + the health card + panels (sessions/usage/context/memory/account).
  A project agent's settings = a MODAL (`agent-detail-view.renderSettingsModal`)
  with the per-agent `agentFields` (name, backend, model, description,
  permission_mode) PATCHing `/api/agents/{id}`. Two different surfaces, two field
  sets.
- **Header** (`web/src/_header.html`): the `.brand` wordmark ("SCUFRIS / scuffed
  jarvis") is a plain `<div>`, NOT a link. The `nav` already has Agent (`/`),
  Agents, Projects, Stats, Settings.
- Much of the RICH settings already exists in pieces: backend->auto-model-default
  + model autocomplete (MB1/F6, `agentFields` + `models_for` + `/api/agents/
  backends`); permission-mode field (B2, `agentFields`); the detailed panels
  (`chat-sidebar.ts` renderContext/renderUsage, plus memory/account renders on the
  settings page). The work is RELOCATING + generalizing these to a per-agent
  settings page, not building them fresh.

## Options considered

### A. Routing/component model

- **A1. Converge components, keep the multi-page shells (RECOMMENDED).** Keep the
  webpack multi-page entries + backend SPA fallback. Make the per-agent chat +
  a new per-agent settings component the ONE implementation; `/` and `/settings`
  mount them with agent id = `orchestrator`, `/agents/<id>` and
  `/agents/<id>/settings` mount them with `agentIdFromPath`. Cheapest: the chat is
  already this shape; only the settings component + a couple of entry wirings are
  new. No client-router dependency.
- **A2. Collapse to a single-page app with a client router.** One HTML shell, a
  JS router mapping `/`, `/settings`, `/agents/<id>[/settings]` to views. Cleanest
  conceptually and kills the entry duplication, but a big rewrite of the
  multi-page setup (6 entries, per-page HtmlWebpackPlugin, the nav's basePath
  templating) for little user-visible gain over A1. Rejected as over-scoped.

### B. Orchestrator as a hidden, projectless, editable agent

- **B1. Hide from the LIST, keep resolvable, make editable via the unified path
  (RECOMMENDED).** `/api/agents` (the list) EXCLUDES the orchestrator by default
  (it is the hidden default); `/api/agents/orchestrator` still resolves so its
  pages work. Lift the per-agent PATCH 409: route `PATCH /api/agents/orchestrator`
  (backend/model/permission_mode) to the SETTINGS STORE (it maps to
  `SCUFRIS_AGENT_BACKEND`/`_MODEL`/... which the orchestrator record already reads)
  so the SAME settings form edits it. Projectless is already true.
- **B2. Leave it in the list but frontend-filter it.** Simpler backend, but the
  "hidden default" is then a frontend concern only and every `/api/agents`
  consumer must remember to filter - leakier. Prefer B1 (authoritative on the
  server) with a frontend guard as belt-and-suspenders.

### C. Unifying the settings surface

- **C1. ONE settings-page component, per-agent, with agent-scoped + shared
  sections (RECOMMENDED).** A single `createAgentSettings(root, {agentId})` page
  that renders: (i) the agent's EDITABLE fields via the existing `agentFields`
  (backend picker -> auto model default, model autocomplete, permission mode,
  description); (ii) the health card; (iii) the detailed panels (context from the
  agent's session; usage/memory/account from the agent's backend account); (iv)
  GLOBAL sections (tools enabled, MCP servers, profiles) shown where they belong
  (see open question); (v) an orchestrator-ONLY extra section for its
  multi-session powers (the session switcher). Mounted as a real PAGE at
  `/settings` (agent=orchestrator) and `/agents/<id>/settings`; the modal is
  retired.
- **C2. Keep two surfaces but share the field builder.** Less churn, but leaves
  the modal-vs-page inconsistency the user explicitly dislikes ("a real /settings-
  style page, not a modal"). Rejected.

## Recommendation

Ship **A1 + B1 + C1**: converge onto the existing multi-page shells; make the
orchestrator a hidden, projectless, editable-through-the-settings-store agent that
is excluded from the `/api/agents` list but resolves at `/api/agents/orchestrator`;
and build ONE per-agent settings PAGE component that both `/settings`
(agent=orchestrator) and `/agents/<id>/settings` mount, retiring the modal. The
chat is already converged, so the routing symmetry is "point the same components
at an agent id". This is additive and low-risk: each piece (auto-model, model
autocomplete, permission mode, the panels, the session switcher) already exists
and is being relocated/generalized, not invented.

Cut into ordered direction-level tasks (backend foundation first so the frontend
has symmetric endpoints to consume):

- **U1 (backend): orchestrator as a first-class, hidden, editable agent.** Exclude
  it from the `/api/agents` list (resolvable at `/api/agents/orchestrator`); lift
  the per-agent PATCH 409 by routing the orchestrator's field edits to the
  settings store; keep it projectless + undeletable. [foundation]
- **U2 (backend): per-agent settings + panel data.** Give every agent id the data
  the settings page needs symmetrically: an effective per-agent config (fields),
  per-agent context (its session), and usage/memory/account dispatched by the
  agent's backend/home - a per-agent analog of the singular endpoints, so the
  frontend fetches one shape for any id. [foundation]
- **U3 (frontend): the unified settings PAGE component.** `createAgentSettings`
  rendering the agent's editable fields + health + detailed panels for ANY agent;
  replaces both `settings-view.ts` and the per-agent modal. [depends U1,U2]
- **U4 (frontend): routing + entries.** `/settings` mounts the unified settings
  with agent=orchestrator; `/agents/<id>/settings` mounts it with `agentIdFromPath`
  (backend catch-all already serves the shell); drop the modal; confirm `/` and
  `/agents/<id>` already share the chat. [depends U3]
- **U5 (frontend): polish + hidden-default UX.** Header wordmark -> link to `/`;
  hide the orchestrator from the `/agents` list (frontend guard over U1's exclude);
  the orchestrator-only multi-session section on its settings page; nav tidy.
  [depends U1,U3]

## Open questions

- **Global vs per-agent settings.** `agent_enabled`, `agent_tools_enabled`,
  `mcp_servers`, `profiles` and `auth_mode` are GLOBAL (the MCP tool server + auth
  are shared across all agents), while backend/model/permission_mode/description
  are PER-AGENT. Should the global sections appear on EVERY agent's settings page
  (editing the shared config) or ONLY on the orchestrator's (it is the "system"
  agent)? Recommend: per-agent pages show per-agent fields + panels + a read-only
  link to the shared/global sections; the orchestrator's settings page hosts the
  editable global sections. Resolve at /plan for U3 (a one-section placement
  decision, not an architecture fork).
- **Orchestrator edits -> settings store mapping.** Confirm the exact mapping
  (`backend`->`agent_backend`, `model`->`agent_model`/`claude_model`,
  `permission_mode`-> is the orchestrator always manual today?) and that a
  rebuild-key change (backend) still clears its session (the existing
  `_on_settings_change`). Pin in U1.
- **`/agents/<id>/settings` deep-link + nav active state.** The nav "Settings"
  link points at `/settings` (orchestrator). A per-agent settings page is reached
  from the agent's own page, not the top nav - confirm the nav active-state logic
  (`nav.ts` `startsWith`) does not mis-highlight. Minor, handle in U4/U5.

## Next steps

Direction-level tasks seeded under umbrella 20260721-234126 (each to be `/plan`ned
into steps when picked up):

- tatr 20260721-234126 (umbrella): Agents UX v3
- tatr 20260721-234558 (U1, backend): orchestrator as a first-class hidden,
  editable agent (exclude from list, edit via settings store)
- tatr 20260721-234609 (U2, backend): per-agent settings + panel data endpoints
- tatr 20260721-234621 (U3, frontend): unified settings PAGE component
- tatr 20260721-234632 (U4, frontend): routing/entries so / == orchestrator and
  /agents/<id>[/settings] share the components
- tatr 20260721-234644 (U5, frontend): wordmark link + hide orchestrator from the
  list + multi-session section + nav tidy

## Fix record

(Each implementing task appends a few lines here as it lands.)
