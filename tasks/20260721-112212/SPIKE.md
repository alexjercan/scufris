# Spike: Agents UX v2 - how to reshape agents into cards + a per-agent chat page

- DATE: 20260721-112212
- STATUS: RECOMMENDED
- TAGS: spike, agents, ux, frontend, backend

## Question

The v1 orchestrator (landed: A0-A5) models an agent as a list row you launch a
one-shot "goal" on. The operator wants a different shape: agents as CARDS that
open a dedicated `/agents/<id>` PAGE with a real CHAT, permission MODES instead
of a write boolean, a cleaner backend surface, an optional description, a unified
orchestrator, and `sesh`-style project discovery. This spike pins the design for
each so the tasks are mechanical, and settles the operator's five open decisions.

## Context

Grounded in the v1 code + an out-of-context UX review of `agents-view.ts` /
`projects-view.ts`. Load-bearing facts:

- The web app is a MULTI-PAGE static bundle: webpack emits `agents/index.html`,
  `projects/index.html`, ... served by FastAPI `StaticFiles(html=True)` with NO
  dynamic path segment and NO SPA fallback - so `/agents/<id>` 404s today.
- The AGENTS page is a pure `renderAgents` + `startAgents` + injected
  `AgentActions`; escaping and race guards are solid. It is a list + a
  detail-panel-below, not cards + a page.
- The backend already has: `AgentStore` (CRUD), `AgentBackend` (codex/claude,
  `stream`/`read_status`, resumes by `session_id`), the run engine
  (`/api/agents/{id}/run|status|events`), and `PATCH /api/agents/{id}` (all
  fields). Backends resume a session, so a chat turn is "stream an arbitrary
  message resuming the agent's session" - the seam exists.
- The LANDING orchestrator runs a SEPARATE path: `CodexCliAgent` + `AgentHandle`
  + the singleton `/api/chat/*` and `/api/agent/session/*` endpoints (list /
  new / switch / fork / delete sessions). Project agents use `AgentBackend` +
  `AgentStore`.
- `AgentRecord` has `goal`/`task_id` but NO `description`. `model` is stamped
  from `settings.agent_model` ("gpt-5.5") for EVERY backend - hence the claude
  "gpt-5.5" bug. Write is a boolean (`write_enabled`) mapped to codex
  `--sandbox` (read-only|workspace-write, first turn only) / claude
  `--permission-mode`.
- `sesh` here is a custom tmux-sessionizer: it lists dirs one level under
  `~/personal ~/personal/_tests ~/work ~/third-party` and `-c <name>` does
  `mkdir ~/personal/<name>` + `tmux new-session -ds`. NO real `sesh` tool.

## Decisions locked (operator, 20260721)

1. **Routing: real dynamic routes + SPA fallback** (not query-string). A
   catch-all in `app.py` serves the agent-detail SPA shell for `/agents/<id>`
   (and `/agents/<id>/settings`), with the client reading the id from the path.
2. **Backend surface: `Codex` and `Claude` only.** `get_backend("codex")` ->
   `CodexBackend("app_server")`; `get_backend("claude")` -> `ClaudeBackend`.
   `exec` is DROPPED from the user surface. `mock` stays DEV-ONLY behind a flag
   (`SCUFRIS_ENABLE_MOCK_BACKEND`, default off) - resolvable + shown only when
   the flag is on. Persisted records store the friendly id (`codex`/`claude`).
3. **Permission MODES (Claude-style), default `manual`.** Replace the
   `write_enabled` boolean with a `permission_mode` enum `manual | edit | auto`:
   - `manual` = read-only (codex `--sandbox read-only`; claude read-only
     posture) - the default.
   - `edit` = may edit files in the project (codex `--sandbox workspace-write`;
     claude `--permission-mode acceptEdits`).
   - `auto` = edit + run commands unattended (codex `workspace-write` with
     approvals never - already the case; claude `--permission-mode
     bypassPermissions`-equivalent).
   The exact per-backend flag for each mode is VERIFIED LIVE before wiring
   (lesson `probe-runtime-on-target-host-early`, now x3) - the mapping above is
   the intent, the flags are confirmed against `codex`/`claude --help` first.
4. **`sesh.py` creates a directory only - NO tmux.** Projects surface DISCOVERED
   dirs (scan the base dirs, one level deep; base dirs configurable, default the
   sesh set) UNION registered projects, inferring metadata (name from dirname,
   language guessed from marker files e.g. pyproject.toml -> python, package.json
   -> node, Cargo.toml -> rust). "Create" = mkdir at the chosen base + register.
5. **Orchestrator unification lands AFTER the per-agent chat** (build order
   below), so the chat component is already parameterized and both paths
   converge onto it.

## Recommendation (design per area)

- **Backend model surface (Task B1).** A `BACKENDS` registry maps friendly id ->
  runner: `codex` -> `CodexBackend("app_server")`, `claude` -> `ClaudeBackend`,
  `mock` -> `MockBackend` (only if `settings.enable_mock_backend`). `get_backend`
  rejects unknown/disabled. `KNOWN_BACKENDS` for the store becomes
  `{codex, claude}` (+ mock when the flag is on). Per-backend DEFAULT MODEL:
  codex -> `settings.agent_model`; claude -> a claude default (e.g. `""` shown as
  "(Claude default)" or an operator-set `settings.claude_model`). Back-compat:
  map legacy persisted `app_server`/`exec` -> `codex` on load. Friendly LABELS
  live in one map (`common.ts` + a backend-name helper) so the UI never shows raw
  ids.

- **Permission modes (Task B2).** `AgentRecord.permission_mode: manual|edit|auto`
  replaces `write_enabled`. `AgentBackend.stream(..., permission_mode=...)`
  replaces `write_enabled=`; each backend maps the mode to its flags (codex
  sandbox level; claude permission mode). Migrate persisted `write_enabled` ->
  `edit` if true else `manual`. Default `manual`.

- **Description + no required goal (Task B3).** Add `description: str = ""` to
  `AgentRecord`/`AgentCreate`/`AgentUpdate` + `common.ts`. The run/chat prompt
  comes from the chat message, not a stored goal; `goal` is retired from the
  create flow (kept as an optional field or dropped - keep it as optional
  metadata to avoid a hard migration, but remove it from the UI).

- **SPA routing + agent cards + detail page (Tasks F1, F2, F3).**
  - F1 (routing): a FastAPI catch-all under `/agents/` that serves the SPA shell
    (the built agents/detail HTML) for any `/agents/<...>` not matching a static
    asset, so client routing works; add the webpack `agent-detail` entry +
    `historyApiFallback` for `/^\/agents\//`.
  - F2 (cards): lift `card()`/`row()` from `stats-view.ts`; `renderAgents` shows
    a `.cards` grid of agent cards (name, friendly backend, state badge, project,
    live turns/tokens). Card click -> `location.assign("/agents/<id>")`. Fold in
    the small UI bugs: friendly backend labels, SSE-on-select reattach (the
    events endpoint already replays via `Last-Event-ID`), a status `setInterval`
    while a running agent is open, and a "create a project first" empty state.
  - F3 (detail page + settings): `agent-detail.ts` reads the id from the path,
    fetches the agent + status, renders the detail + a per-agent settings form
    (name, description, backend, permission mode) that calls the existing
    `PATCH /api/agents/{id}`. Extract a shared `agentFields()` builder reused by
    create + settings.

- **Per-agent chat (Tasks B4, F4).**
  - B4 (backend): `POST /api/agents/{id}/chat` (a message) streams a turn via
    `get_backend(agent.backend).stream(prompt=message, session_id=agent.session_id,
    cwd=project.cwd, permission_mode=agent.permission_mode)` through the SAME
    supervisor + event bus as `run`, capturing/persisting `session_id`. One
    session per agent (resumed each turn). `GET /api/agents/{id}/transcript`
    reads that session's history (reuse `read_transcript` / the claude session
    reader) so the page can rebuild the conversation.
  - F4 (frontend): the detail page hosts the chat, REUSING the pure helpers from
    `agent-view.ts` (`parseSseFrames`, `sendChatStream`, markdown render,
    no-yank scroll, composer) but de-globalized into `agent-detail.ts`'s own
    state, targeting the per-agent chat/events/transcript endpoints.

- **Orchestrator as a default agent (Task B5).** Model the landing orchestrator
  as a RESERVED agent (fixed id e.g. `orchestrator`, not in `agents.json`,
  undeletable, no project binding required) that routes through `get_backend`
  like the others - this is the deferred "decision 4" plus a special record.
  It KEEPS the multi-session features (`/api/agent/session/*`: new/switch/fork/
  list/delete); project agents are single-session and do NOT expose them. The
  landing page and the per-agent chat converge on the same chat component; the
  orchestrator's page just also shows the session switcher. Land this AFTER F4.

- **sesh.py + Projects discovery (Task B6/F5).** `scufris/sesh.py`: `discover()`
  scans configurable base dirs one level deep -> candidate `{path, name,
  language?}`; `create(name, base)` -> `mkdir` (NO tmux) and returns the path.
  `GET /api/projects` (or a new `/api/projects/discovered`) returns discovered
  UNION registered; create registers + mkdirs. Projects page lists both, marking
  which are registered.

## Open questions

- The EXACT codex sandbox flag and claude permission-mode value for each of
  manual/edit/auto - resolve by `codex --help` / `claude --help` + a live probe
  BEFORE wiring B2 (do not guess; the lesson is now x3).
- Whether to keep `goal`/`task_id` on the record as optional metadata or migrate
  them out - default: keep, hide from UI (cheapest).
- Orchestrator: does it need a project binding at all, or run in the server cwd
  as today? Default: no binding (server cwd), decided in B5.
- Auto-registering a discovered project on first use vs an explicit "register"
  action - decided in F5.

## Next steps

Direction-level tasks this spike seeds (see GOAL.md for the live queue; build
order encodes deps). Each is coarse - `/plan` expands it into steps when the
flow picks it up.

- B1: backend surface cleanup (Codex/Claude only, mock dev-flag, drop exec,
  per-backend default model, friendly labels).
- B2: permission modes (manual|edit|auto) replacing write_enabled, per-backend.
- B3: agent `description` + retire the required goal.
- F1: SPA dynamic routing + fallback + the agent-detail page shell.
- F2: agent cards + friendly labels + the small UI bug fixes (SSE-on-select,
  status interval, empty state).
- F3: `/agents/<id>` detail page + per-agent settings-edit (PATCH).
- B4: per-agent chat endpoint (+ transcript).
- F4: per-agent chat UI (reuse the agent-view chat helpers).
- B5: orchestrator as a reserved default agent (multi-session) - AFTER F4.
- B6+F5: sesh.py discovery + Projects discovery/create (no tmux).

## Fix record

(Filled by each implementing task as it lands.)
