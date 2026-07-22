# EPIC: Agents UX v3 - one agent surface (orchestrator as a hidden default), shared settings + chat, richer settings

- DATE: 20260721
- UMBRELLA TASK: 20260721-234126
- LANDING SCOPE: squash-merge each task to `master` (local default), do NOT push
  (user's call). Standard flow landing.

## Goal

Collapse the landing page and the per-agent pages into ONE agent surface where
the orchestrator is just a special "hidden default" agent, so every agent shares
the same chat + settings UI and only the URL differs. From the user's feedback
(2026-07-21):

1. The header wordmark ("SCUFRIS - scuffed jarvis") is a link back to the landing
   page (`/`).
2. The orchestrator is HIDDEN from the `/agents` list (the "hidden default"
   style), and ideally has NO associated project (it runs in the server cwd).
3. Settings are UNIFIED across all agents: the same settings a user has for the
   orchestrator are available for a project agent and vice versa. Keep the health
   check section. Each agent's settings live at its own `/settings`-style page
   (a real page, not just a modal).
4. The landing page IS conceptually `/agents/orchestrator`, and the main
   `/settings` page IS conceptually `/agents/orchestrator/settings` - so all
   agents share one chat UI and one settings UI, and the orchestrator is merely
   exposed directly at `/` (and `/settings`) instead of under `/agents/...`.
5. Settings improvements: pick the backend and the model auto-defaults to that
   backend's default; change the model via autocomplete; change the sandbox /
   permission mode (manual -> edit -> auto); surface the detailed panels (context
   usage, memory, account, ...) for every agent. The orchestrator MAY get an extra
   section for its multi-session powers.

Direction/feasibility is being pinned in this EPIC's SPIKE (routing model, how to
share the settings surface, orchestrator-as-hidden-default, projectless
orchestrator, per-agent detailed panels). This GOAL.md then holds the task queue
the flow drives.

## Outcome (CLOSED 2026-07-22)

All five tasks landed to master (squash-merged, not pushed): U1 10c54d3, U2
fae9161, U3 47cfc5e, U4 7f96905, U5 9aa0f9a. Final suite on master green: ruff
format + lint, mypy, 282 backend tests, 140 web tests, webpack build; tatr check
--ledger LESSONS.md clean for every task. The user accepted and closed the
umbrella (trusting the automated suite; manual browser eyeballing to follow, any
issues filed as new tasks).

App-feedback follow-ups from the 2026-07-22 review are tracked as separate tasks
(NOT in this umbrella): 20260722-104034 (p60, bug - claude agents show
codex-specific health/settings; make the settings page backend-aware),
20260722-104043 (p40 - /projects/<id> detail page), 20260722-104048 (p30 - broad
terminal-aesthetic styling pass from the kitty config), 20260722-104058 (p5,
ideation - configurable theming). Two feedback items were already delivered in U5
(clickable wordmark; orchestrator settings back-link no longer leaking
/agents/orchestrator).

## Done means

(Refined by the SPIKE; observable acceptance, each naming its proof.)

1. The header wordmark links to `/`. (test: the rendered header anchor href="/";
   manual: clicking it returns to the landing chat.)
2. The orchestrator does not appear in the `/agents` list and has no project
   binding. (test: GET /api/agents / the agents page render excludes the
   orchestrator; the orchestrator record has no project_id.)
3. One shared settings UI serves every agent (orchestrator + project agents) with
   the same fields + the health section, mounted as a real per-agent settings
   PAGE. (test: the settings component renders for a project agent and the
   orchestrator; manual: edits persist across reload.)
4. Routing: `/` and `/settings` render the orchestrator's chat/settings using the
   SAME components as `/agents/<id>` and `/agents/<id>/settings`. (test: routing +
   SPA fallback; manual: `/`, `/settings`, `/agents/<id>`, `/agents/<id>/settings`
   all work.)
5. Settings: backend pick auto-defaults the model; model autocomplete; permission
   mode switch; the detailed panels (context/memory/account) show for every agent;
   the orchestrator shows its extra multi-session section. (test: the API +
   render; manual: the live flows.)

Overall: full check suite green on master (ruff + mypy + pytest + web `npm run
ci`), and `tatr check --ledger LESSONS.md` clean for this EPIC's tasks.

## Tasks

Seeded by SPIKE tasks/20260721-234433/SPIKE.md; the flow's /plan expands each into
steps. Order = build sequence (backend foundation first).

- [x] 20260721-234558 (p50, U1) orchestrator as a first-class hidden, editable agent (exclude from list, edit via settings store) [backend]
      landed 10c54d3; 1 review round (out-of-context APPROVE, 2 NITs no-action). AgentStore.list() hides the orchestrator (resolvable via get()); PATCH /api/agents/orchestrator edits it via the settings store (backend->agent_backend, model->agent_model|claude_model by effective backend, permission_mode->new agent_permission_mode setting; 403 read-only, 422 invalid, backend change clears its session). claude_model + agent_permission_mode added to WRITABLE_KEYS + AgentConfigUpdate (kept in sync); .env.example refreshed. Backend 281 + web 151 green. Lesson bumped: always-present-synthetic-item-invalidates-empty-assertions (x2, now both directions).
- [x] 20260721-234609 (p48, U2) per-agent settings + panel data endpoints (context/usage/memory/account per agent) [backend; dep U1]
      landed fae9161; 1 review round (out-of-context APPROVE, 1 NIT adopted). GET /api/agents/{id}/usage|memory|account resolve the agent (404) + dispatch by backend via _agent_is_codex: real codex-account data for a codex agent + the orchestrator, None/empty for claude/mock (no reader - honest). Context reuses the existing /status (no dup endpoint). Backend green; no web change. Lesson: assert-a-distinct-value-not-the-default.
- [x] 20260721-234621 (p46, U3) unified settings PAGE component for all agents (replaces settings-view + the modal) [frontend; dep U1,U2]
      landed 47cfc5e; 1 review round (out-of-context APPROVE, 3 adopted + 2 deferred). New agent-settings-view.ts (createAgentSettings + pure renderAgentSettings): shared agentFields form + reused Health card + context/usage/memory/account panels, mounted at /agents/<id>/settings via the shell's path-branch (agent-detail.ts); the per-agent modal + its dead CSS retired, sidebar Settings is now a link (orchestrator too). Web 154 tests green; no backend change. Deferred: per-agent/claude-aware health (R3), read-only wiring (R5) -> U4. Lesson bumped: render-rewrite-orphans-its-css (x2).
- [x] 20260721-234632 (p44, U4) routing/entries so / == orchestrator and /agents/<id>[/settings] share the components [frontend; dep U3]
      landed 7f96905; 2 review rounds (out-of-context REQUEST_CHANGES on a read-only MAJOR, then APPROVE). /settings and /agents/orchestrator/settings render the SAME createAgentSettings (shared agentSettingsDeps); the orchestrator's global sections (System toggles/MCP/tools/profiles) fold in via settings-view's now-exported renders (backend/model once). Retired settings-view's renderSettings/startSettings + ~250 dead lines; writable flows from config.writable (read-only server -> read-only view, no live-but-403 controls). Web green (frontend-only). Lesson bumped: moving-logic-off-a-scope-drops-its-incidental-guarantees (x2).
- [x] 20260721-234644 (p42, U5) hidden-default polish - wordmark link, hide orchestrator from list, multi-session section, nav [frontend; dep U1,U3]
      landed 9aa0f9a; 1 review round (out-of-context REQUEST_CHANGES on an ASCII arrow + polish, all adopted). The SCUFRIS wordmark is now an <a href=basePath> (returns to the landing chat); renderAgents filters the reserved orchestrator out of the /agents grid (belt-and-suspenders over U1's server exclude) + its dead no-delete guard removed; an orchestrator-only Sessions panel (count + current + a link to /) fed by /api/agent/sessions shows only for the orchestrator; its settings back-link points at / so /agents/... never leaks. Web 140 tests + build green. Lesson: grep-touched-files-for-non-ascii-before-commit.

## Manual acceptance (batched for the user at Finish)

- U3: `/agents/<id>/settings` is a real PAGE (not a modal) for a project agent -
  fields + Health + context/usage/memory/account panels; the Settings affordance
  on the agent page links to it.
- U4: `/settings` and `/agents/orchestrator/settings` render the SAME page; the
  orchestrator's global sections (System toggles / MCP / tools / profiles) appear
  there and NOT on a project agent's; backend/model appear once; edits persist
  across reload; a read-only server (SCUFRIS_SETTINGS_WRITABLE=0) shows a read-only
  view (no live-but-403 controls).
- U5: clicking the "SCUFRIS / scuffed jarvis" wordmark returns to `/`; the
  orchestrator is absent from the `/agents` grid; the orchestrator's settings page
  shows a Sessions section (count + current + a "manage on the chat" link); its
  back-to-chat link goes to `/` (not `/agents/orchestrator`).
