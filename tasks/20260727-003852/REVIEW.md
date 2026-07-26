# Review: harden den_path Settings test against the dev .env

- TASK: 20260727-003852
- BRANCH: (none - test-only hotfix on master)

## Round 1

- VERDICT: APPROVE
- REVIEWER: in-session (trivial test-isolation diff; A/B verified)

The diff makes the `_enabled()` test helper pass `_env_file=None` so the agent
tests' "nothing set" baseline ignores the repo `.env`. A/B verified: with the
operator's `.env` (SCUFRIS_DEN_PATH=~/personal/the-den) present,
`test_scufris_mcp_server_injects_den_path_for_orchestrator_only` was RED before the
change and the full suite is 524 passed after; `ruff` and `mypy` clean (the
pydantic-settings `_env_file` init arg carries a scoped `# type: ignore[call-arg]`,
since it is not in the generated model signature). `nix flake check` was green both
before and after (its sandbox has no `.env`). No findings. APPROVE.
