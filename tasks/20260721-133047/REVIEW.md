# Review: MB1 agent model follows backend + editable model in settings

- TASK: 20260721-133047
- BRANCH: fix/model-follows-backend

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Both suites ran green in the worktree: backend (ruff + mypy 35 files + pytest all
pass) and frontend (npm run ci, 153 tests + webpack build). Zero findings.

Verified by the reviewer:
- `AgentStore.update()` traces correctly for all five cases: backend change with
  no model re-defaults; explicit non-empty model wins; blank model re-defaults;
  model-only edit keeps the model and backend; same-backend PATCH leaves the
  model untouched. `backend_changed` compares the canonicalized incoming backend
  vs the load-canonicalized `agent.backend`, so a legacy `app_server` id does
  not spuriously count as changed.
- `GET /api/agents/backends` is declared before `/api/agents/{agent_id}` (not
  shadowed, confirmed by the passing endpoint test) and respects
  `enable_mock_backend` (mock only under the dev flag).
- `claude_model` defaults to `claude-opus-4-8`, still overridable by
  `SCUFRIS_CLAUDE_MODEL`; no code still assumes the old empty default.
- Frontend: the shared `agentFields(context, backends, initial)` seam is intact;
  the model input auto-fills on a backend `change`; the create form sends
  `model` and the settings form PATCHes it; change-updates-model tests dispatch a
  real event and assert the new default. Dead `AGENT_BACKENDS` removed cleanly.
- `test_update_backend_redefaults_model` would fail on the old update(); no
  tests were weakened.

No BLOCKER/MAJOR/MINOR/NIT issues. APPROVE.
