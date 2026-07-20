# Retro: Settings backend - editable tools (per-tool disable + MCP add/remove)

- TASK: 20260720-184137
- BRANCH: feature/settings-editable-tools
- REVIEW ROUNDS: 1 (APPROVE; 1 MINOR + 1 NIT fixed in-round)

## What went well

- Probed the integration before building: `codex mcp list -c
  mcp_servers.x.env.KEY=...` confirmed codex accepts per-server env, so the
  "pass disabled-tools to the server via env, filter at startup" design was
  known-good before any code (probe-runtime-on-target-host lesson paid off).
- Found the real enforcement point: codex registers whole MCP servers, so the
  guard had to live INSIDE the scufris server (`apply_disabled_tools` ->
  `remove_tool`), not in the UI or the agent. The `enabled` API flag is a
  mirror, not the guard - and the tests assert the registry, not just the API.
- The review's LOW finding (trailing-newline id via `re.match`+`$`) was a real
  latent hole; fixing it with `re.fullmatch` and unifying the regex across the
  two boundaries closed a genuine drift risk.

## What went wrong

- Strict-validator detour: added id/command validators to `McpServerSpec`,
  which broke the existing `test_mcp_overrides_skips_invalid_or_reserved_id`
  (it deliberately constructs a bad-id spec) and would have crashed startup on
  a bad env entry. Reverted and moved validation to the endpoint. Root cause:
  tightened the model before reading its existing tests/usage.
- `re.match` + `$` accepted `"fs\n"` at both boundaries - the classic Python
  anchor gotcha; caught by review, not by me.
- The out-of-context reviewer used non-canonical severities (LOW/INFO), which
  failed `tatr check` after landing; had to remap to MINOR/NIT.

## What to improve next time

- Before tightening a shared model's validation, grep its existing tests and
  construction sites - a permissive-by-design model often has a skip elsewhere.
- Use `re.fullmatch` (or `\A...\Z`) for id/whole-string validation; `^...$`
  admits a trailing newline.
- Constrain review subagents to the four canonical severities in the prompt so
  their REVIEW.md passes `tatr check` without a remap.

## Action items

- [x] Lessons added: codex per-server env for tool filtering; fullmatch for id
      validation; review-severity constraint (see LESSONS.md).
- No follow-up code task.
