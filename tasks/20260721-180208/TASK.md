# B5bc: retire the Agent protocol + move orchestrator sessions to the unified model

- STATUS: OPEN
- PRIORITY: 33
- TAGS: agents,backend

## Story

MERGED B5b + B5c (originally split by the recon, but reading the code shows they
are architecturally inseparable): `CodexCliAgent.current_session_id()` is the
SINGLE session state shared by BOTH the landing chat endpoints (post_chat /
post_chat_stream / post_chat_reset) AND the session endpoints (get_sessions /
post_session switch|new / fork_session / get_context / delete session). Retiring
the `Agent` protocol forces moving that session state, and the session endpoints
depend on it - so chat-rerouting and session-migration must land together.
Splitting them would need a throwaway session-holder shim.

Goal: the orchestrator runs entirely through `get_backend(...).stream()` + the
supervisor (like project agents), with its multi-session state living in the
agent model (the store), and the `Agent` protocol / `CodexCliAgent` /
`AgentHandle` / `build_agent` retired.

## Coarse steps (/plan expands when picked up)

- [ ] Store: give the orchestrator real session management - the active session
      (`_orch_session_id`, from B5a) plus `set_orchestrator_session(id|None)` for
      switch/new; the session LIST comes from per-backend discovery (codex
      `list_sessions`; a claude equivalent) behind the backend interface.
- [ ] Reroute the landing chat endpoints to the orchestrator record: streaming
      `post_chat_stream` -> `_launch_agent_turn(orchestrator, None, msg)`;
      non-streaming `post_chat` -> collect the backend stream's final reply;
      `post_chat_reset` -> clear the orchestrator session.
- [ ] Reroute the session endpoints to the store's orchestrator session +
      per-backend discovery (get_sessions, switch/new, fork = new + seed turn,
      get_context, delete). These stay ORCHESTRATOR-only (project agents are
      single-session and 404/hide them).
- [ ] Preserve `/api/agent/info`/`config`/`context`/`usage`/`account`/`health`
      by sourcing from the orchestrator record + backend read_status/read_context
      (or re-point), and keep `agent_backend`/`agent_model` as the config source.
- [ ] Retire `Agent` protocol, `CodexCliAgent`, `AgentHandle`, `build_agent`, and
      the Agent-shaped `MockAgent`/`DisabledAgent` once nothing imports them (the
      disabled-state gate moves to a settings check; Mock is a BackendMock). Update
      every test that mocks `Agent` (there are many).

## Definition of Done

- The landing/orchestrator chat runs through `get_backend(...).stream()` via the
  supervisor; no `Agent`-protocol code remains
  (cmd: `grep -rn "class CodexCliAgent\|Agent(Protocol)\|class AgentHandle" scufris/` -> gone).
- The orchestrator keeps multi-session (list/switch/new/fork/delete) on the
  unified model; project agents do not expose it
  (test: multi-session only for the reserved id; a landing/orchestrator chat is a
  supervised run).
- Full check suite green (backend `pytest -q` + web `npm run ci`).
- manual: hold a multi-turn landing/orchestrator conversation, switch sessions,
  and it all still works end to end.

## Notes
- Depends on: B5a (20260721-112439, landed). Blocks: B5d.
- HIGHEST-RISK slice of the EPIC (retires an abstraction, moves session state,
  reroutes ~10 endpoints, rewrites many `Agent`-mocking tests). Probe the
  exec/app_server + claude session-resume semantics before wiring; land in
  reviewable pieces if needed.
- Per-backend session DISCOVERY differs (codex rollouts vs claude session files);
  keep it behind the `AgentBackend` interface so it is not codex-shaped.
- Absorbs the old B5c (20260721-180219), which is closed as merged.
