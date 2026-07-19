# Agent backend: multi-session registry + context & usage/quota endpoints

- STATUS: CLOSED
- PRIORITY: 30
- TAGS: feature, agent, backend, spike

## Goal

The data layer for the agent-page expansion. Generalize the agent's single
`_thread_id` into a settable current-session id, and expose three things codex
already records on disk: the list of sessions (to switch between), a per-session
context object (window size + token usage + tool-call counts), and the account's
usage/quota (weekly rate limit). This unblocks the sidebar and the context/usage
panel.

Likely surface (for `/plan` to refine): list sessions from
`$CODEX_HOME/sessions/**/*.jsonl` (line-1 `session_meta` + first `user_message`
title; filter to this app's sessions), `CodexCliAgent` gains
`switch_session(id)` / `new_session()` over its current-id, and endpoints
`GET /api/agent/sessions`, `POST /api/agent/session`, `GET /api/agent/context`,
`GET /api/agent/usage`. Usage/context come from the latest `token_count` payload
(`info.model_context_window`, `info.total_token_usage`, `rate_limits.primary`
with `window_minutes: 10080` = weekly, `used_percent`, `resets_at`, `plan_type`).

## Decisions (from /plan, resolving the spike's open questions)

- Read from the codex ROLLOUT files, not the exec `--json` stdout. Session
  listing needs the rollouts anyway, and context/usage live in the same
  `token_count` payload there; one source, robust across a restart. (Harvesting
  the live stdout `token_count` is a later optimization, not needed now.)
- Scope the session list by `cwd == <server cwd>` AND `originator == "codex_exec"`
  - this matches codex's own default resume filter and keeps unrelated codex
  sessions (other dirs, interactive TUI) out of the app's list.
- New module `scufris/sessions.py` holds the rollout parsing + models, keeping
  `agent.py` about driving codex. Functions take an explicit `codex_home: Path`
  and `cwd: str` so tests drive them by writing fake rollout `.jsonl` into a tmp
  home (integration-style, per AGENTS.md) with no codex binary.
- Session methods go on the `Agent` protocol (implemented as no-ops/None in
  `DisabledAgent`) so `create_app` needs no isinstance checks.

## Steps

- [x] `scufris/sessions.py`: models `RateWindow` (used_percent, window_minutes,
      resets_at), `UsageQuota` (plan_type, primary, secondary), `SessionInfo`
      (id, title, started_at, updated_at, git_branch, cwd), `SessionContext`
      (session_id, context_window, input/cached/output/reasoning/total tokens,
      turn_count, tool_call_count). `resolve_codex_home(settings) -> Path`
      (settings.codex_home | `$CODEX_HOME` | `~/.codex`).
- [x] `scufris/sessions.py`: `list_sessions(codex_home, cwd) -> list[SessionInfo]`
      - glob `sessions/**/*.jsonl`, parse line-1 `session_meta` (id, cwd, ts,
      git.branch), take the first `user_message.message` as the title (bounded,
      escaped later by the UI), filter by cwd+originator, sort newest-first;
      `read_context(codex_home, session_id) -> SessionContext | None` (locate the
      rollout by id, scan for the LAST `token_count` -> window + total usage, and
      count `user_message` turns + `mcp_tool_call_end` tools);
      `read_usage(codex_home) -> UsageQuota | None` (newest rollout carrying a
      `token_count` -> its `rate_limits`). Defensive line parsing (skip malformed),
      bounded reads.
- [x] `scufris/agent.py`: generalize `CodexCliAgent._thread_id` to a current
      `_session_id`; add `current_session_id()`, `new_session()`,
      `switch_session(id)`; extend the `Agent` protocol with these three and
      implement in `DisabledAgent` (None / no-op). Keep `reset()` = `new_session()`.
- [x] `scufris/app.py`: `GET /api/agent/sessions` -> `{sessions, current}`,
      `POST /api/agent/session` (body `{action: "new"|"switch", session_id?}` ->
      set current, return `{current}`; 503 when the agent is disabled),
      `GET /api/agent/context` -> `SessionContext | null` (current session),
      `GET /api/agent/usage` -> `UsageQuota | null`. GET endpoints degrade to
      empty/null when disabled. Resolve codex_home + cwd from settings/os.
- [x] Tests: `tests/test_sessions.py` (write fake rollout `.jsonl` into a tmp
      codex_home: list filters by cwd/originator + title + sort; read_context
      token/turn/tool counts; read_usage rate_limits; malformed lines skipped).
      Extend `tests/test_agent.py` (new/switch/current over a fake runner) and
      `tests/test_app.py` (the four endpoints with a fake agent + tmp codex_home,
      incl. the disabled-agent degradation + 503 on POST).
- [x] `nix develop` full check green (ruff, ruff format, mypy, pytest) and a live
      smoke: run the real endpoints against this host's `$CODEX_HOME` and confirm
      `/api/agent/sessions` lists real sessions and `/api/agent/usage` returns the
      weekly window.

## Definition of Done

- The four endpoints work: sessions can be listed and switched/new-ed, the
  current session's context (window + token usage + turn/tool counts) is exposed,
  and the account usage/quota (weekly `used_percent` + `resets_at` + plan) is
  exposed - all read from codex rollouts, degrading cleanly when the agent is off.
  Session filtering keeps unrelated codex sessions out. `ruff`/`mypy`/`pytest`
  green; live-verified against the real `$CODEX_HOME`.

## Implementation

- New `scufris/sessions.py`: models `RateWindow`/`UsageQuota`/`SessionInfo`/
  `SessionContext`; `resolve_codex_home`; `list_sessions(home, cwd)` (glob rollout
  heads, filter by cwd+originator, title from first user_message, sort by mtime);
  `read_context(home, id)` (locate rollout by id in the filename, last
  `token_count` -> window+usage, count `user_message` turns + `mcp_tool_call_end`
  tools); `read_usage(home)` (newest rollout's `rate_limits`). Defensive line
  parsing; client `session_id` is `glob.escape`d before rglob.
- `scufris/agent.py`: `Agent` protocol gains `current_session_id`/`new_session`/
  `switch_session` (impl in `CodexCliAgent` over a generalized `_session_id`, and
  no-op/None in `DisabledAgent`); `reset()` -> `new_session()`.
- `scufris/app.py`: `GET /api/agent/sessions` ({sessions, current}),
  `POST /api/agent/session` (new/switch; 503 disabled, 422 switch-without-id),
  `GET /api/agent/context`, `GET /api/agent/usage`. GET degrade to empty/null off.
- Tests: `test_sessions.py` (fake rollouts: cwd/originator filter, sort, title,
  context counts, usage window, malformed-line + glob-escape guards); extended
  `test_agent.py` (switch/new/current) and `test_app.py` (the four endpoints +
  disabled degradation). Live-verified against the real `~/.codex`: 3 sessions,
  weekly window 10080 min, context window 258400.

## Notes

- Spike: tasks/20260719-212152/SPIKE.md (the probe that confirmed the data is on
  disk; the "context" breakdown is intentionally codex's real axes, not a
  per-component split codex does not expose).
- Blocks tatr 20260719-212205 (sidebar) and 20260719-212207 (context/usage panel).
- Keep the read-only, injectable-seam patterns; parse rollouts defensively (skip
  malformed lines) as `_parse_events` already does.
