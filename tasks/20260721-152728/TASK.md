# F5: agent detail UX reshape - chat-first + stats sidebar (no sessions) + Settings modal

- STATUS: OPEN
- PRIORITY: 39
- TAGS: agents,frontend

## Story

User feedback (2026-07-21): entering an agent should GREET you with the chat,
not a settings form. Reshape `/agents/<id>` to mirror the landing agent page:

- The CHAT is the primary surface (top / center), shown first.
- A stats sidebar on the left like the main agent page, but WITHOUT the
  "Sessions" box (a project agent has exactly one session).
- The detailed settings card moves BEHIND a "Settings" button that opens it
  (a modal / drawer / disclosure), where you can switch backend/model/mode/etc.
  (the F3 editable form).

## Steps

- [ ] Restructure `agent-detail.html` + `startAgentDetail`/`startAgentChat` so the
      layout is chat-primary with a left stats sidebar. Keep the chat in its own
      root (F4 lesson `persistent-widget-needs-its-own-root-not-a-polled-region`).
- [ ] Reuse the landing page's stat boxes from `agent-view.ts` where they fit
      (context window / usage / account) but OMIT the Sessions box. Verify what
      the per-agent `/status` already carries vs. what the landing boxes need.
- [ ] Move the F3 settings form into a "Settings" disclosure/modal opened by a
      header button; closing returns to the chat. Keep the read-only project +
      live state badge in the header.
- [ ] Per-agent status poll updates the stats sidebar (focus-guarded), never
      wiping the chat or a mid-edit settings form.
- [ ] Tests: chat-first on load; Settings button toggles the form; stats sidebar
      renders turns/tokens with NO Sessions box; existing detail tests updated.

## Definition of Done

- Opening `/agents/<id>` shows the chat immediately (no click)
  (test: chat present on load; settings hidden until the button is clicked).
- A stats sidebar shows live turns/tokens/context, no "Sessions" section
  (test: stats present, no sessions box).
- The Settings button opens the editable card and edits still persist (F3)
  (test: toggle reveals the form; save dispatches).
- Full web gate passes (cmd: `npm run ci` in web/).
- manual: the page feels like the landing agent (chat-first, stats left,
  settings a click away).

## Notes
- Depends on: F1 (routing), F3 (settings form), F4 (chat). All landed.
- Reuse map: `web/src/agent-view.ts` has `renderContext`/`renderUsage`/
  `renderSessions` (SKIP sessions) + stat-box helpers; `agent-detail-view.ts`
  (settings+status) and `agent-chat-view.ts` (chat) are the current modules.
- Decide at /plan: modal vs drawer vs inline disclosure for Settings - simplest
  that keeps the chat mounted.
