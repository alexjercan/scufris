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

Seeded by the SPIKE (below); the flow's /plan expands each into steps.

- (pending SPIKE)

## Manual acceptance (batched for the user at Finish)

- (accumulates as tasks land)
