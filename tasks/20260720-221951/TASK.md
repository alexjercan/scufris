# A4: Agents dashboard page (live status list; fold Projects into agent creation)

- STATUS: OPEN
- PRIORITY: 22
- TAGS: spike,agents,frontend

## Goal

The Agents dashboard page: the LIST view polls `GET /api/agents` for coarse
live status (state, last activity, tokens), reusing the Stats page polling +
client-side sparkline patterns and the pure-render + injected-actions seam. The
FOCUSED/open agent view uses SSE (`GET /api/agents/{id}/events`) relayed from the
supervisor event bus (ADR-001) for live token streaming - drop-safe, replays on
reconnect. Fold the standalone Projects page into the agent-creation flow
(project becomes a picker). This is what turns the AGENT/STATS gimmicks into a
real cockpit.

## Steps

- [ ] `common.ts`: add `Agent` (mirror `AgentRecord`: id, name, project_id,
      backend, model, goal, task_id, session_id, state, write_enabled) and
      `AgentRunStatus` (agent_id, state, session_id, turns, tool_calls,
      input/output_tokens, context_window, last_message, updated_at) interfaces.
- [ ] `agents-view.ts`: a PURE `renderAgents(root, agents, projects, selectedId,
      status, actions)` (jsdom-testable, no fetch) - an agent list with a state
      badge + backend + project per row; a create form (name, a PROJECT PICKER
      <select> over existing projects, a backend <select>, a goal textarea, a
      write-enabled checkbox); and a detail panel for the selected agent
      (metadata, the polled `AgentRunStatus`, a Run button, a live events log
      area, delete). Escape every host/user string. `startAgents` does the fetch
      orchestration: load agents + projects; on select, poll `/api/agents/{id}/
      status`; Run -> `POST /api/agents/{id}/run` then open an SSE `EventSource`
      on `/api/agents/{id}/events` appending frames to the log (race-guarded like
      the projects tasks fetch).
- [ ] `agents.html` + `agents.ts` (thin entry: `initNav(); void startAgents();`),
      a `agents` webpack entry + `HtmlWebpackPlugin` (chunks:["agents"],
      filename agents/index.html) + a `historyApiFallback` rewrite for
      `/^\/agents/`.
- [ ] Nav: add an "Agents" link in `_header.html` (first, as the primary
      cockpit). Fold-note: the project picker in agent creation is the "fold";
      the Projects page stays for project CRUD (removing it is a separate task).
- [ ] `style.css`: agent list/badge/detail rules (reuse settings/projects
      classes; add `.agents*` where needed, state-colored badges).
- [ ] vitest jsdom tests (`agents-view.test.ts`): renders the agent list +
      create form (with the project picker options); selecting an agent shows its
      detail + status; a hostile agent name/goal is escaped; create submits the
      form values; the Run button calls the run action.
- [ ] Verify end to end: `cd web && npm run ci`, then serve the built bundle
      through the backend and confirm `/agents/` lists/creates an agent and shows
      status (`frontend-verify-needs-e2e-serve`).

## Definition of Done

- `renderAgents` shows the list + create form (project picker) and a selected
  agent's detail + status (test: `agents_page_renders_list_and_detail`).
- A hostile agent name/goal injects no markup
  (test: `agents_page_escapes_hostile_strings`).
- The Run button dispatches the run action (test: `agents_page_run_dispatches`).
- `npm run ci` passes in `web/` (cmd: `cd web && npm run ci`).
- manual: load `/agents/`, create an agent bound to a project, run it, and see
  its status/state update.

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (recommendation 5).
- Depends on: 20260720-221929 (A1), 20260720-221942 (A3, landed 263a769).
- Reuse the Projects page pattern (projects-view.ts): pure render + injected
  actions, `sendJson` mutations, single authoritative render (reload after a
  mutation), the select-race guard.
- Live events use `EventSource('/api/agents/{id}/events')` (GET SSE) - simplest
  for a GET stream; the SSE wiring lives in `startAgents` (not the pure render),
  so jsdom tests cover render only and the stream is e2e-verified.
- Lessons: `webpack-multipage-htmlplugin-per-page`, `type-change-fails-strict-tsc-not-vitest`
  (run full `npm run ci`), `side-effect-free-module-for-jsdom-tests`,
  `escape-only-host-strings-in-element-content`; symlink `web/node_modules` into
  the worktree, NEVER `git add -A`.
