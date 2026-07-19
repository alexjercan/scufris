# Agent page: left sidebar with session list and switching

- STATUS: OPEN
- PRIORITY: 20
- TAGS: feature, agent, ui, spike

## Goal

Restructure the agent (landing) page into a claude.ai / chatgpt-style two-pane
shell: a left sidebar with a "new chat" button and a list of sessions, and the
existing chat in the main pane. Clicking a session switches the agent to it
(retarget resume; optionally re-render that session's past transcript). New chat
starts a fresh session.

Likely surface (for `/plan`): `index.html` two-column layout + CSS; sidebar
renders `GET /api/agent/sessions` (title + time), highlights the current one,
"new chat" and per-session click call `POST /api/agent/session`. Decide during
/plan whether switching re-renders history (parse the session transcript) or just
retargets - the spike flags this as a nice-to-have the backend session object can
support.

## Decisions (from /plan)

- **Re-render history on switch.** A sidebar that switches to a blank chat pane is
  the shim the flow warns against - it is not the claude.ai experience the user
  asked for. So this task includes a small backend transcript reader + endpoint so
  clicking a session shows its conversation, and continuing it works (codex
  resumes the same session server-side). This is the one place the sidebar needs
  a bit of backend, and it is inseparable from a good sidebar.
- **New-chat moves into the sidebar** (claude.ai placement); the head "new chat"
  button is removed and its reset logic re-points at the sidebar button.
- The weekly-usage meter is NOT here - it belongs to tatr 20260719-212207; this
  task leaves the sidebar's lower slot for it.

## Steps

- [ ] `scufris/sessions.py`: `TranscriptMessage {role, text}` +
      `read_transcript(codex_home, session_id, limit) -> list[TranscriptMessage]`
      (user_message -> user, agent_message final_answer -> assistant; bounded;
      reuses the glob-escaped `_find_rollout`).
- [ ] `scufris/app.py`: `GET /api/agent/session/{session_id}` ->
      `{messages: [...]}` (empty when the agent is off). Extend backend tests
      (`test_sessions.py` read_transcript; `test_app.py` the endpoint).
- [ ] `web/src/common.ts`: `SessionInfo`, `SessionsResponse`, `TranscriptMessage`
      types (mirror backend).
- [ ] `web/src/index.html`: wrap the chat in a two-column `.agent-shell` - a left
      `.sidebar` (a "+ new chat" button + `#session-list`) and the existing
      `.chat` main pane; drop the head new-chat button.
- [ ] `web/src/agent-view.ts` (keep pure helpers exported for jsdom):
      `renderSessions(sessions, currentId)` (list items: title + relative time,
      active highlight, escaped titles), `switchSession(id)` (POST switch -> fetch
      + render transcript -> reset usage state -> re-highlight), `newChat()` (POST
      new -> clear log -> reset -> refresh list). `startAgent` loads the session
      list; after each reply, refresh the list so a new session appears with its
      title and the current one stays highlighted.
- [ ] `web/src/style.css`: `.agent-shell` grid, `.sidebar`, `.sidebar__new`,
      `.session` (+ `.is-active`), responsive stack on narrow screens. Themed.
- [ ] `web/src/agent-view.test.ts`: `renderSessions` (items, active highlight,
      hostile-title escaped, empty state). LIVE serve smoke: sidebar lists real
      sessions, clicking one loads its transcript, "new chat" clears; `npm run ci`
      + `ruff`/`mypy`/`pytest` green.

## Definition of Done

- The agent page is a two-pane shell: a left sidebar lists the agent's sessions
  (newest first, current highlighted) with a "+ new chat" button; clicking a
  session loads its past transcript and continues it; new chat starts fresh.
  Session titles escaped; render stays side-effect-free for jsdom; jsdom +
  `npm run ci` + python checks green; serve-verified with real sessions. Stats
  page untouched.

## Notes

- Spike: tasks/20260719-212152/SPIKE.md.
- Depends on tatr 20260719-212203 (sessions endpoints - CLOSED).
- Keep the render module side-effect-free for jsdom, escape all host/session
  strings (titles come from user messages), theme it like the rest.
