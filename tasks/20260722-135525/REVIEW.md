# Review: Add the opencode backend (opencode serve -> llama.cpp)

- TASK: 20260722-135525
- BRANCH: feature/opencode-backend

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Out-of-context reviewer ran every proof independently: full pytest + the two
named test files (20 new tests) PASS; `ruff check .` PASS; `mypy scufris/` clean;
`mypy .` = 44 == master's 44 (zero net-new, debt tracked in 20260722-153555); a
LIVE end-to-end turn through `get_backend("opencode")` against the warm daemon
returned a real gemma reply with read_status/read_transcript reading it back.
Verified the sync-vs-async boundary is sound (the read handlers are sync `def`,
run in a threadpool, so the blocking httpx read never touches the event loop) and
that the test-assertion updates are genuine, not masking a break. In-session
supplement re-verified the sync-handler call sites (app.py:1122/1231,
mcp_server.py:288 are all sync `def`).

- [x] R1.1 (MINOR) scufris/config.py:209 - comment "The two user-facing backends
  are 'codex' and 'claude'" is stale now that opencode is a third user-facing
  backend.
  - Response: fixed - comment updated to "codex, claude and opencode".
- [x] R1.2 (NIT) scufris/backends.py `_read_messages` docstring - the "asyncio.run
  would raise inside the running loop" rationale is misleading (the callers are
  sync threadpool handlers, not on the loop).
  - Response: fixed - docstring now explains the callers are sync `def`
    threadpool handlers, naming the call sites; approach unchanged (it was
    already correct).
- [ ] R1.3 (NIT) frontend test fixtures (web/src/agent-fields.test.ts:104 and the
  `backends()` fixtures) list only codex/claude. Not a bug - the picker is
  data-driven from /api/agents/backends and common.ts already carries the
  opencode + "local" labels; adding opencode to a fixture only broadens coverage.
  - Response: left as-is (optional coverage; runtime is correct and proven).
- [ ] R1.4 (NIT) TASK step said `SCUFRIS_OPENCODE_AUTH_MODE` but the field is
  `agent_opencode_auth_mode` -> env `SCUFRIS_AGENT_OPENCODE_AUTH_MODE` (what
  .env.example documents). Implementation is the correct/consistent choice
  (mirrors `agent_claude_auth_mode`); task text was imprecise.
  - Response: acknowledged; no code change - the implemented env var is the
    intended, consistent one.

Pending user (manual) checks: the manual DoD (a turn through the backend returns
a coherent reply from gemma-4-26B-A4B-it) was confirmed live by both the
implementer and the reviewer ("backend works"). Nothing left open.
