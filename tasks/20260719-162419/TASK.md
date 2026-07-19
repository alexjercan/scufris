# Build Scufris MCP server: curated agent tools (tatr_*, host_stats)

- STATUS: OPEN
- PRIORITY: 12
- TAGS: feature,backlog,agent,tools,security

## Goal

Build the Scufris MCP server exposing a curated, allowlisted set of tools
(`tatr_*`, `host_stats`) to the agent, backed by safe subprocess handlers,
registered with Codex under `[mcp_servers.scufris]`.

## Notes

- Spike: tasks/20260719-153050/SPIKE.md (recommends a single stdio MCP server;
  allowlist is the set of handlers; never a shell string).
- Each tool = one typed Python handler (pydantic args) -> `subprocess.run([...],
  shell=False, timeout=...)` with captured/bounded output, returning a structured
  result. `host_stats` reuses the existing `Collector` (tatr 20260719-154420) -
  no second host-data source.
- Start read-only (`tatr_ls`, `tatr_show`, `host_stats`); add mutating tools
  (`tatr_new`, `tatr_edit`) deliberately, gated by Codex's approval policy.
- Choose the MCP Python lib (official `mcp` SDK vs `fastmcp`) during /plan.
- Safety is the headline (AGENTS.md): allowlist only, no arbitrary shell,
  validate every argument, timeouts + output caps.
- Test handlers in isolation (call with args, assert result) without the LLM.
- Depends on the agent backend (tatr 20260719-162356) for how Codex is launched
  and configured. The same MCP server works for any MCP-capable harness.
