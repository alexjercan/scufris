# Review: B1 backend surface cleanup

- TASK: 20260721-112429
- BRANCH: refactor/backend-surface

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Suites (reviewer ran both): backend ruff+mypy clean, 253 passed; frontend
`npm run ci` green, 135 passed. Verified in-session.

Reviewer verified the risky edges: canonical_backend folds app_server/exec/codex
-> codex (unknown left as-is so get_backend rejects); get_backend RESOLVES mock
always while the store GATES create/update by the flag (a persisted mock agent
still runs if the flag is later off); legacy records normalize on load and
persist on next write (nothing dropped); per-backend default model fixes the
claude "gpt-5.5" bug; helpers in config avoid an agent_store->backends import (no
cycle). Tests are meaningful (gating, claude-model, legacy-normalize, get_backend
failure case all discriminate their mechanism).

- [ ] R1.1 (MINOR) settings-view.ts BACKENDS still shows raw app_server/exec/mock
  for the PROCESS chat agent's `agent_backend` field (a separate Settings field,
  out of B1's per-agent scope). Two backend vocabularies now coexist; reconcile
  in F2/B5.
  - Response: Tracked - added a carried-in note to B5 (20260721-112439) to
    reconcile the settings picker when the orchestrator becomes an agent. Not
    fixed here (a genuinely separate field, out of B1 scope).
- [ ] R1.2 (NIT) test name drift (`test_backend_canonicalized_and_claude_default_model`
  vs the DoD's `claude_agent_default_model`) - assertion correct, name differs.
  - Response: Left - the test is present and meaningful; DoD name was aspirational.
- [x] R1.3 (NIT) config.py redundant paren `else (settings.agent_model)`.
  - Response: Fixed - rewrote default_model_for as a plain if/return.
- [ ] R1.4 (NIT) agents-view raw `agent.backend` display - deferred to F2 labels.
  - Response: Confirmed deferred to F2 (friendly labels); no stored-value-not-in-
    picker bug since new agents only pick codex/claude and the detail is display-only.

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (a one-line cosmetic simplification of default_model_for;
  no behavior change)

Verification: `default_model_for` rewritten as if/return; ruff + mypy clean, 253
backend + 135 frontend still pass. No new findings.
