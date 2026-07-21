# Review: U2 - per-agent usage/memory/account panel endpoints

- TASK: 20260721-234609
- BRANCH: feature/per-agent-panel-data

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (fresh subagent, no sight of the implementing session;
  ran the backend suite itself and re-derived the dispatch correctness +
  no-missed-claude-reader + no-enabled-gate questions)

Clean backend-only diff. Suite green: `ruff` + `mypy` (20 files) + `pytest`
(~290, all pass); `git diff --stat master...HEAD` shows 0 web/ changes. Verified:
`_agent_is_codex` uses `canonical_backend` (legacy app_server/exec count as codex);
all three endpoints `_require_agent` (404) BEFORE dispatch; the orchestrator
resolves them; `account.model` is the agent's effective model; the memory-empty
shape is byte-identical to the singular endpoint; no route shadowing; and there is
genuinely NO claude usage/footprint reader in the tree, so None/empty for non-codex
is honest, not a stub. The deliberately-omitted `agent_enabled` gate is correct (a
project agent has no enabled flag; the global orchestrator toggle must not blank a
project agent's panels).

- [x] R1.1 (NIT) tests/test_app.py - the account assertion only checked
  truthiness; because the codex agent was created with no model, its model equaled
  the global default, so the test could not catch an `account.model=
  settings.agent_model` bug. Create the codex agent with an explicit distinct
  model and assert on it.
  - Response: Fixed. The codex agent is now created with `model=gpt-5-codex-custom`
    and the test asserts `account["model"] == "gpt-5-codex-custom"` - it now fails
    if account returned the global model. Confirmed green.
- [ ] R1.2 (NIT) scufris/app.py `agent_account` - `enabled` is the global
  `settings.agent_enabled` for every agent. Mirrors the singular endpoint + the
  AccountInfo model; defensible/consistent.
  - Response: No change - consistent with the singular /api/agent/account and the
    shared AccountInfo model; the account is the codex account backing the agent.

No pending manual DoD items (backend-only; DoD machine-proved).
