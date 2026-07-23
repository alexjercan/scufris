# Review: wire scufris MCP into the claude backend (20260723-201851)

- DATE: 20260723
- REVIEWER: out-of-context agent (general-purpose), round 1
- COMMIT: aa97452
- VERDICT: APPROVE

## Summary

Clean, well-scoped implementation. The shared-core refactor is behaviour-preserving
for codex, the claude formatter is correct, and the design decisions (whole-server
wildcard allowlist, `--strict-mcp-config` tradeoff) are sound and documented. Every
hard-scrutiny item verified; no blocking or non-blocking correctness issues.

## Findings

### Blocking
None.

### Non-blocking
None.

### Nits (optional)
1. `backends.py` `_claude_stream_args` docstring: "Pure (a filesystem lookup, no
   subprocess)" now also reads `settings.host`/`port` via `scufris_mcp_server`.
   Still pure (no side effects), but "a filesystem lookup" undersells it.
   -> ADDRESSED in follow-up commit (reworded to "a filesystem lookup + a settings
   read").
2. `agent.py` `scufris_mcp_server` docstring wording ("or a regular agent turn with
   no id") - accurate as written; no change needed.

## Verified (hard-scrutiny items)

- Variadic `--mcp-config` boundary: safe in EVERY ordering. Its value is always
  immediately followed by `--strict-mcp-config`; the only trailing block is
  `--resume <id>` (flag+value, never a bare positional). JSON is one argv token via
  `create_subprocess_exec` (no shell). Pinned by tests.
- Whole-server `mcp__scufris__*` allowlist is role-safe: `mcp_server.main()` calls
  `apply_role` which REMOVES out-of-role tools from the live registry before serving;
  the allowlist governs auto-approval, not exposure.
- Codex behaviour unchanged: env content byte-identical; all edge cases preserved
  (tools disabled -> []; no role but enabled -> operator servers + approval_policy;
  orchestrator wins over agent_id).
- `--strict-mcp-config` dropping project/operator MCP for claude: documented tradeoff
  (DECISION.md + TASK.md), additive since claude had zero MCP wiring before.
- `_agent_has_scufris_mcp` -> codex OR claude: only gates the scoped-tools endpoint;
  the codex-only `_agent_is_codex` (usage/memory/account) is untouched.
- Tests match the implementation; the mock-agent case still exercises "no wiring -> []".
