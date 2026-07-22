# Review: backend-aware health

Out-of-context review of the branch (feature/backend-aware-health) against the
DoD. Reviewer read the actual diff, not the narrative.

- VERDICT: APPROVE (one MINOR test-strengthening adopted)

## Findings

- MINOR: no test proved a POPULATED `backend_version` came from the right backend
  (the existing cases used missing bins -> None on both branches). Adopted: added
  `test_agent_health_backend_version_comes_from_the_probed_backend` - a fake
  executable claude bin emits "claude 9.9.9"; the claude probe populates
  `backend_version` with it while `backend == "claude"` and `claude cli` is ok.
- NIT: `canonical_backend(backend or settings.agent_backend)` is belt-and-
  suspenders (agent records + the settings enum are already canonical). Kept -
  harmless, future-proofs a legacy id. No change.
- NIT (scoped out, informational): the account panel's `auth mode` still renders
  the global codex-flavored `auth_mode`; the app does not model claude auth. This
  was explicitly out of scope (see the TASK DoD boundary); captured as a follow-up
  below. No change here.

## Verified clean (reviewer)

- A claude agent truly gets claude checks: `agent.backend == "claude"` ->
  `effective_backend == "claude"` -> the claude branch probes `claude_bin` and sets
  `backend_version` from claude `--version`. Confirmed by the app + unit tests.
- `canonical_backend` is applied, so legacy ids (`app_server`/`exec`) fold to
  `codex` and do not fall through to a no-probe path.
- `backend_version` is never populated for the wrong backend (assigned only inside
  each backend's own block).
- The orchestrator resolves via `_require_agent` (synthetic record); the test
  asserts `backend == "codex"` and `"codex_version" not in` the response.
- No regression to the global `/api/agent/health` (unchanged). No orphaned
  `codex_version` remains. The Python `AgentHealth` and the TS interface match.
- The frontend test genuinely proves the URL switch (asserts the per-agent
  `/api/agents/builder/health` is requested and the global one is not).
- No newly introduced non-ASCII typography.

## Follow-up surfaced (not this task)

- The account panel's `auth_mode` is still codex/ChatGPT-flavored for every agent.
  If the user wants claude auth surfaced, model claude's auth and dispatch the
  account panel by backend (mirrors this health fix).
