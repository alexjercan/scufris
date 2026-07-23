# DECISION: backend-agnostic MCP-config core, per-backend formatters

- TASK: 20260723-193218 (spike)
- DATE: 20260723
- STATUS: ACCEPTED

## Context

Codex wires the scufris MCP server via `agent._mcp_overrides` (`-c
mcp_servers.scufris.*` config overrides + auto-approve). The claude backend needs
the SAME server (same command, args, role env, scoping) but a DIFFERENT format
(`--mcp-config` JSON + `--allowedTools`). The live spike proved claude's surface
works end to end.

## Options

1. COPY the role env/command logic into a standalone `_claude_mcp_config` in
   `backends.py`. Fast, but two copies of the role->env mapping drift over time (the
   exact failure T2 avoided by extracting `role_tool_names`).
2. EXTRACT a backend-agnostic core `mcp_server_config(settings, *, is_orchestrator,
   agent_id) -> {command, args, env, tool_names}` that both backends FORMAT: codex
   to `-c` overrides, claude to `--mcp-config` JSON + `--allowedTools`. One source of
   truth for "which server, which env, which role tools"; each backend owns only its
   wire format.

## Decision

Option 2. The CONTENT (server command, role env, tool scoping) is identical across
backends and is already the scoping source of truth; only the FORMAT differs. A
shared core is the same discipline as `role_tool_names` (T2) and keeps codex and
claude from advertising different surfaces for the same role. Codex keeps its
exact current `-c` output (guarded by existing tests); claude adds its formatter.

## Consequences

- `_mcp_overrides` becomes a thin codex formatter over the core; its output must
  stay byte-identical (existing codex tests pin this).
- A new claude formatter builds the `{"mcpServers":{...}}` JSON + `--allowedTools`
  (`mcp__scufris__<tool>`), bounded so the variadic `--mcp-config` cannot eat later
  argv tokens.
- `_agent_has_scufris_mcp` (app.py, T2) generalizes to include claude once wired, so
  the role-scoped tools view updates itself with no UI change.
- Implemented in task 20260723-201851.
