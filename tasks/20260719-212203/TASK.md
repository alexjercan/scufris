# Agent backend: multi-session registry + context & usage/quota endpoints

- STATUS: OPEN
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

## Notes

- Spike: tasks/20260719-212152/SPIKE.md (the probe that confirmed the data is on
  disk; the "context" breakdown is intentionally codex's real axes, not a
  per-component split codex does not expose).
- Blocks tatr 20260719-212205 (sidebar) and 20260719-212207 (context/usage panel).
- Keep the read-only, injectable-seam patterns; parse rollouts defensively (skip
  malformed lines) as `_parse_events` already does. Resolve the open questions in
  the spike (stdout token_count vs rollout; cwd vs originator filter) during /plan.
