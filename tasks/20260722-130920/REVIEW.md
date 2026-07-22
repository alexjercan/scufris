# Review: account auth_mode backend-aware

Out-of-context review of the branch (feature/account-auth-backend-aware) against
the DoD. Reviewer read the actual diff and ran the frontend suite.

- VERDICT: APPROVE (no findings)

## Verified clean (reviewer)

- Correctness: `auth_mode_for_backend` folds legacy codex ids via
  `canonical_backend` (pinned by the config test app_server -> chatgpt). All FOUR
  original `settings.agent_auth_mode` reporting sites in app.py are converted with
  the right dispatch backend (per-agent AccountInfo by `agent.backend`;
  AgentInfo/AgentConfig/global AccountInfo by `settings.agent_backend`). No site
  missed. The one BEHAVIORAL use (agent.py:462, the codex `login --with-api-key`
  gate) is correctly left as `settings.agent_auth_mode`.
- Type/contract: the three Python models are `AuthMode | None`; the three TS
  interfaces are `string | null`. Both frontend reads (account panel + config row)
  route through `authLabel`, which always returns a string, so `configRow` still
  gets a non-null value and null is handled.
- Test quality: the per-agent account test proves codex -> chatgpt, claude ->
  claude_ai, mock -> None on distinct backends via the real endpoint (would fail on
  revert; the claude assertion proves it is not the codex value). The api_key
  overrides are exercised per backend. The frontend tests prove the label map, the
  null/unknown handling, and that the raw `claude_ai` wire value is NOT shown.
- Regressions: `test_account_endpoint_shape` (codex -> chatgpt) still holds. The
  new `agent_claude_auth_mode` is correctly NOT in WRITABLE_KEYS/AgentConfigUpdate
  (not per-agent writable) - no parity concern.
- Honesty: `agent_claude_auth_mode` is a declared/effective value with the same
  status as the codex one; the boundary (no claude login flow / ANTHROPIC wiring)
  is respected in code, comments, and .env.example.
- No non-ASCII introduced.
