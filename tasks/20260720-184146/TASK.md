# Settings backend: console data endpoints (memory footprint + account)

- STATUS: CLOSED
- PRIORITY: 35
- TAGS: feature,agent,backend

## Story

As the operator, I want the console to show me informative panels about what
the agent is doing and holding: its sessions, usage/quota, current context,
its persistent "memory" footprint, and the account backing it. Most read-only
data endpoints exist; this task adds the two that do not (memory footprint and
a consolidated account view) so the frontend (task 6) has data to render.

## Steps

- [x] Defined "memory" as the agent's persistent footprint:
      `sessions.MemoryFootprint` + `read_memory_footprint` (rglob rollout
      count, total bytes, oldest/newest mtime; empty footprint when the dir is
      missing) and `GET /api/agent/memory`. Never raises.
- [x] `GET /api/agent/account` = `AccountInfo{auth_mode, model, enabled,
      quota}` reusing `read_usage`. No cheap codex account identity beyond auth
      mode was available, so it is omitted (noted below).
- [x] Confirmed existing `/usage`, `/context`, `/sessions` already cover their
      panels - no duplication; account reuses `read_usage`, memory is the one
      genuinely new datum.
- [x] Tests: `test_memory_endpoint_reports_footprint`,
      `test_memory_endpoint_empty_ok`, `test_memory_zero_when_disabled`,
      `test_account_endpoint_shape`, `test_account_quota_null_when_disabled`.

## Definition of Done

- `GET /api/agent/memory` reports session count and total size for a known
  temp rollout dir, and empty/zero (not an error) when the dir is missing
  (test: `memory_endpoint_reports_footprint`, `memory_endpoint_empty_ok`).
- `GET /api/agent/account` returns auth mode, model and the usage quota
  (test: `account_endpoint_shape`).
- Full suite green (cmd: `nix develop --command bash -c "ruff check . && mypy . && pytest -q"`).

## Notes

- Relevant files: `scufris/sessions.py` (`_sessions_dir`, `list_sessions`,
  `read_usage`, `read_context`), `scufris/app.py` (existing usage/context/
  sessions endpoints ~339-366), `scufris/health.py`.
- No hard dep on the writable tasks - these are read-only additions; can land
  in parallel with the backend store work.
- "memory" is deliberately the rollout footprint, not a new memory system;
  a richer agent-memory concept would be its own spike.

## Close-out

- Small, purely-additive read-only task: one new datum (memory footprint) +
  one consolidation (account = auth_mode/model/enabled + reused `read_usage`
  quota). Both endpoints degrade to empty/zero when the agent is disabled or
  the codex sessions dir is missing, mirroring `agent_health`'s never-raise
  contract, so the frontend (T6) can render them unconditionally.
- `read_memory_footprint` scans `rollout-*.jsonl` (the same glob `read_usage`
  uses) for count/bytes/oldest/newest; a per-file `stat()` OSError is skipped,
  not fatal.
- Account identity: codex exposes no cheap account handle beyond the auth mode
  (the rollouts carry usage but not an account name), so `AccountInfo` omits a
  name field rather than shelling codex for it - can revisit if a source turns
  up. This is the deliberate "omit if not cheaply available" from the plan.
- No frontend here (T6 renders these panels). No dep on the writable store.
