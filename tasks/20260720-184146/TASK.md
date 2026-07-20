# Settings backend: console data endpoints (memory footprint + account)

- STATUS: OPEN
- PRIORITY: 35
- TAGS: feature,agent,backend

## Story

As the operator, I want the console to show me informative panels about what
the agent is doing and holding: its sessions, usage/quota, current context,
its persistent "memory" footprint, and the account backing it. Most read-only
data endpoints exist; this task adds the two that do not (memory footprint and
a consolidated account view) so the frontend (task 6) has data to render.

## Steps

- [ ] Define "memory" concretely as the agent's PERSISTENT footprint: add
      `GET /api/agent/memory` returning the codex rollout footprint - session
      count, total on-disk size, oldest/newest timestamps (from the sessions
      dir scufris already scans in `scufris/sessions.py`). Read-only; never
      raises (mirror `agent_health`).
- [ ] Add `GET /api/agent/account` consolidating the account view: auth mode,
      model, and the usage/quota window (reuse `read_usage`); include any
      account identity codex exposes if cheaply available, else omit. Read-only.
- [ ] Confirm the existing `GET /api/agent/usage`, `/context`, `/sessions`
      already return what the panels need; note any gap in Notes rather than
      duplicating.
- [ ] Tests: memory endpoint reports counts/size against a temp sessions dir
      with known rollouts; account endpoint returns auth mode + model + quota;
      both never raise when the agent is disabled or the dir is missing.

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
