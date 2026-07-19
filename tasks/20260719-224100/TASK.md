# Agent sessions: delete a conversation

- STATUS: OPEN
- PRIORITY: 35
- TAGS: feature, agent, ui, spike

## Goal

Let the user delete a session from the sidebar. Backend: a `delete_session(codex_
home, session_id)` that removes the codex rollout file for that id (via the
glob-escaped `_find_rollout`, unlink), and a `DELETE /api/agent/session/{id}`
endpoint (503 when the agent is off). If the deleted session is the current one,
reset the agent to a new session. Frontend: a small delete (x) affordance on each
session row with a confirm, that removes it and refreshes the list (and clears the
chat if it was the active one).

## Notes

- Spike: (none - direct feature request). Builds on the sessions backend
  (tatr 20260719-212203) and the sidebar (20260719-212205).
- Deleting only ever unlinks the one validated rollout file inside CODEX_HOME; it
  is destructive, so require an explicit confirm in the UI and validate the id
  server-side (reuse the `_SERVER_ID_RE`-style guard / `_find_rollout`).
- Tests: `delete_session` removes the file and is a no-op for an unknown id;
  the endpoint deletes + resets-current-when-matching + 503 when disabled;
  frontend renderSessions gains a delete control (jsdom).
- Depends on nothing else in this batch; can flow first or second.
