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

## Notes

- Spike: tasks/20260719-212152/SPIKE.md.
- Depends on tatr 20260719-212203 (sessions endpoints).
- Keep the render module side-effect-free for jsdom, escape all host/session
  strings (titles come from user messages), theme it like the rest. Stats page
  untouched.
