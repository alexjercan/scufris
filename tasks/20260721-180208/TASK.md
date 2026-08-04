# B5bc: retire the Agent protocol + move orchestrator sessions to the unified model

- PRIORITY: 33
- TAGS: agents, backend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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

- [x] Store: give the orchestrator real session management - the active session
      (`_orch_session_id`, from B5a) plus `set_orchestrator_session(id|None)` for
      switch/new. (Session LIST still uses the codex `list_sessions` disk
      discovery; per-backend claude discovery stays a follow-up note.)
- [x] Reroute the landing chat endpoints to the orchestrator record: streaming
      `post_chat_stream` -> `_launch_agent_turn(orchestrator, None, msg,
      image_paths, on_done=cleanup)` relayed via `_relay_bus_sse`; non-streaming
      `post_chat` -> `_drain_turn(bus).reply`; `post_chat_reset` -> clear the
      orchestrator session.
- [x] Reroute the session endpoints to the store's orchestrator session
      (get_sessions, switch/new, fork = clear + seed turn, get_context, delete).
      Serialize on `ORCHESTRATOR_ID` (not the old "chat" key). Fork launches its
      seed turn WITHOUT the outer serialize lock (that would self-deadlock on the
      same key that `_launch_agent_turn` reserves).
- [x] Preserve `/api/agent/info`/`config`/`context`/`usage`/`account`/`health`:
      they already source from settings + codex home, so no rewiring needed;
      `agent_backend`/`agent_model` stay the config source.
- [x] Retire `Agent` protocol, `CodexCliAgent`, `AgentHandle`, `build_agent`, and
      the Agent-shaped `MockAgent`/`DisabledAgent`. The disabled-state gate is now
      a `settings.agent_enabled` check in each endpoint; the CLI `chat` drives
      `get_backend(...).stream()`; the settings-store backend-switch clears the
      orchestrator session (preserving the cross-backend-stale-session fix).
      Rewrote the test touchpoints (FakeAgent -> FakeBackend via get_backend
      monkeypatch + `app.state.agents` seeding; deleted the retired-class tests).

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

## Close-out (landed, all-or-nothing reroute)

Foundation (`91706bc`, prior session): `set_orchestrator_session`/
`orchestrator_session_id` on the store + `image_paths`/`on_done` on
`_launch_agent_turn`. This session completed the reroute + retirement:

- create_app dropped the `agent: Agent | None` param, the `AgentHandle`/
  `build_agent` wiring, and now exposes `app.state.agents`. The settings-store
  `on_change` clears the orchestrator session when `agent_backend` changes
  (keeps the cross-backend-stale-session fix, previously carried by the handle).
- Landing chat rerouted: `post_chat` drains `_launch_agent_turn`'s bus for the
  reply, `post_chat_stream` relays it via `_relay_bus_sse` (with image temp +
  reconnect), `post_chat_reset` clears the session. Session endpoints reroute to
  the store's orchestrator session and serialize on `ORCHESTRATOR_ID`.
- Retired from agent.py: `Agent`, `DisabledAgent`, `MockAgent` (+ its mock
  helpers), `CodexCliAgent`, `build_agent`, `AgentHandle`. The exec/app-server
  runners + parse fns stay. CLI `chat` now drives `get_backend(...).stream()`.
- Tests: `FakeAgent` -> a rich `FakeBackend` injected by monkeypatching
  `scufris.app.get_backend`; session state seeded via `app.state.agents`;
  deleted the retired-class tests in test_agent / test_settings_store / test_cli.

Bug hit + fixed during the reroute: fork **self-deadlocked**. It held
`supervisor.serialized(ORCHESTRATOR_ID)` and then called `_launch_agent_turn`,
which reserves the SAME serialize key inside `supervisor.start` - a nested
acquire of a non-reentrant per-key lock that hung forever (caught by bisecting
the ~30%-mark hang to `test_fork_seeds_new_session_with_prior_context`). Fix:
fork launches its seed turn WITHOUT the outer lock; `_launch_agent_turn` already
serializes and 409-guards, and the set-then-launch is synchronous so nothing
interleaves. Lesson `serialize-then-launch-self-deadlocks-on-shared-key`.

Green: backend `ruff` + `mypy` + `pytest` (EXIT=0); web `npm run ci` (168 tests).
DoD grep `grep -rn "class CodexCliAgent\|Agent(Protocol)\|class AgentHandle"
scufris/` -> empty.

## Notes
- Depends on: B5a (20260721-112439, landed). Blocks: B5d.
- HIGHEST-RISK slice of the EPIC (retires an abstraction, moves session state,
  reroutes ~10 endpoints, rewrites many `Agent`-mocking tests). Probe the
  exec/app_server + claude session-resume semantics before wiring; land in
  reviewable pieces if needed.
- Per-backend session DISCOVERY differs (codex rollouts vs claude session files);
  keep it behind the `AgentBackend` interface so it is not codex-shaped.
- Absorbs the old B5c (20260721-180219), which is closed as merged.
