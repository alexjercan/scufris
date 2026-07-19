# Review: Agent chat panel in the dashboard

## Round 1 - 20260719

Scope: `scufris/agent.py` (thread continuity + reset), `scufris/app.py` (chat
endpoints), `web/src/{index.html,main.ts,style.css}`, `tests/test_agent.py`,
`tests/test_app.py`.

### Correctness

- The whole feature is PROVEN live on this host: `/api/chat` turn 1 sets a
  codeword, turn 2 recalls it (`BANANA` - real multi-turn continuity via
  `codex exec resume`), and `/api/chat/reset` -> the next turn answers `unknown`
  (fresh conversation). This is the real DoD, not just green units.
- The resume-sandbox gotcha was caught by the live run and fixed: `codex exec
  resume` inherits the session sandbox and rejects `--sandbox`, so it is only
  passed on turn 1. Good that this surfaced before landing.
- Concurrency is handled: chat turns (and reset) run under an `asyncio.Lock`, so
  overlapping requests can't interleave codex sessions or race the thread id.
- `AgentUnavailable` -> HTTP 503 with the message; the panel renders a clear
  disabled state from `/api/config` `agent_enabled`. App still serves stats/panel
  with the agent off.
- XSS-safe in the chat path: messages are inserted with `textContent`, not
  `innerHTML`, so agent/user text can't inject markup (unlike the stat cards,
  tracked separately in 20260719-160924).
- Tests: backend reply/503/reset, continuity + reset via the fake runner, and the
  fake-codex integration test now emits `thread.started` and asserts the parsed
  id. ruff + mypy + pytest + `npm run ci` all green.

### Observations (non-blocking)

- LOW: one global conversation is shared across browser tabs/clients (the agent
  is a single app-instance object). Correct for a single-user local dashboard;
  worth noting if multi-client is ever wanted.
- LOW: after a page reload the DOM log is empty but the backend still holds the
  codex thread, so the next message resumes the pre-reload context. Minor UX
  surprise; "new chat" resets it. Fine for v1.
- NOTE: replies are turn-based (pending bubble -> full reply), not
  token-streamed, because `codex exec --json` emits turn-level events only.
  Documented honestly in the task rather than faked.

### Verdict

APPROVE. Meets the Definition of Done - a working, themed chat panel with real
multi-turn continuity and reset, verified live returning GPT-5.5 replies;
serialized backend endpoints; disabled/error states; green checks including a
real subprocess test. The LOW items are single-user-appropriate and noted.
