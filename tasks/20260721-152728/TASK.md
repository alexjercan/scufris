# F5: agent detail UX reshape - chat-first + stats sidebar (no sessions) + Settings modal

- PRIORITY: 39
- TAGS: agents, frontend
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Design (locked from the reuse map)

- Two-pane `.agent-shell` grid (`260px 1fr`): LEFT `#agent-sidebar` (`.sidebar`),
  RIGHT `#agent-chat` (the F4 chat, its own root - primary/greeting surface).
- Sidebar (top-to-bottom): back link, header (agent name + live state badge), a
  "Settings" button, then stat boxes (`.usage-block`): a "Status" box (state,
  turns, tools) and a "Context" box (context-window bar + input/output tokens),
  both fed by `GET /api/agents/<id>/status`.
- SKIP the Sessions box (one session per agent) and, for now, the account/usage
  quota box (needs a per-backend account endpoint - flagged in Notes, deferred).
- Settings behind a button: clicking "Settings" opens a MODAL overlay holding the
  F3 editable form (name/backend/model/desc/mode + save); closing returns to
  chat. The modal is a SEPARATE root, so the status poll (which re-renders the
  sidebar) never wipes a mid-edit form.

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

- [x] `agent-detail.html`: two-pane `.agent-shell` shell - `#agent-sidebar`
      (`.sidebar`) + `#agent-chat` (chat, own root) + a hidden
      `#agent-settings-modal` overlay root + keep `#agent-detail` removed/folded.
- [x] Rewrite `agent-detail-view.ts`: split the old `renderAgentDetail` into
      (a) pure `renderSidebar(root, agent, project, status, actions)` - back link,
      name + state badge, a "Settings" button (calls `actions.openSettings`), a
      "Status" `.usage-block` (state/turns/tools) and a "Context" `.usage-block`
      (context-window bar + input/output tokens); reimplement the tiny
      row/bar helpers locally (reuse the CSS classes, not agent-view internals);
      and (b) `renderSettingsModal(root, agent, project, backends, actions)` - the
      F3 settings form (unchanged) inside a modal card with a close button.
- [x] `startAgentDetail()`: fetch agent/project/backends/status; render the
      sidebar; wire the Settings button to render + show the modal (and close to
      hide it); poll `/status` -> re-render the sidebar only. The modal is a
      separate root so a mid-edit form survives the poll.
- [x] `agent-detail.ts`: keep `startAgentDetail()` + `startAgentChat()`.
- [x] `style.css`: `.agent-shell`/`.sidebar`/`.usage-block` already exist (reuse);
      add a small `.agent-modal` overlay (fixed, backdrop, centered card) + the
      sidebar Settings button. Ensure the chat pane fills the right column.
- [x] Tests: rewrite `agent-detail-view.test.ts` for `renderSidebar` +
      `renderSettingsModal` (chat-first: sidebar has no form until Settings is
      clicked; a Status/Context box exists; NO sessions box; the modal's form
      still saves; back link + not-started + XSS on name preserved).

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
- Decision: MODAL overlay for Settings (native, no lib) - keeps the chat mounted
  and reads as a "Settings button opens the options" per the feedback.
- DEFERRED (flagged): an account/usage-quota box on the sidebar. The landing
  `/api/agent/usage` is the CODEX landing account, wrong for a claude agent; a
  correct per-agent box needs a per-backend account/quota endpoint. Left out of
  F5; revisit if wanted (small backend task). The Status + Context boxes deliver
  the "stats on the left" ask from data the per-agent `/status` already carries.
- `AgentRunStatus` lacks `cached_input_tokens`/`reasoning_output_tokens`, so the
  Context box omits those rows (shows context-window %, input, output).
- Close-out: split `renderAgentDetail` into pure `renderSidebar` (header + state
  badge + Settings button + Status/Context `.usage-block`s) and pure
  `renderSettingsModal` (the F3 form inside an overlay card with close +
  backdrop-click). The shell is a two-pane `.agent-shell` grid (sidebar + chat)
  plus a hidden `#agent-settings-modal` overlay; `startAgentDetail` renders the
  sidebar, wires the Settings button to render+show the modal, and polls
  `/status` -> re-render sidebar only (the form lives in the separate modal root,
  so a mid-edit survives the poll). Reused the landing `.agent-shell`/`.sidebar`/
  `.usage-block`/`.bar` CSS; added `.agent-modal` with the `[hidden]` guard (the
  flex-defeats-hidden lesson). Stat helpers reimplemented locally (kvRow/statBox)
  rather than coupling to agent-view internals. Bundle-verified the built shell
  carries `#agent-sidebar`/`#agent-chat`/`#agent-settings-modal`; the interactive
  chat-first flow is the batched manual check. 163 frontend tests.
