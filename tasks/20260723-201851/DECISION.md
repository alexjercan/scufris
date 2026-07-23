# DECISION: claude MCP allowlist uses a whole-server wildcard, not enumerated names

- DATE: 20260723
- TASK: 20260723-201851
- STATUS: ACCEPTED

## Context

Wiring the scufris MCP server into the claude backend needs an `--allowedTools`
entry so the unattended turn auto-approves the scufris tools instead of hanging
on an approval prompt (lesson
`claude-mcp-tool-approval-is-allowedTools-not-permission-mode`). The open
question from the spike (Step 3): does claude accept a whole-server wildcard
(`mcp__scufris__*`), or must every tool name be enumerated - the orchestrator
role exposes ~17 tools, the agent role only `request_input`.

## Decision

Use a single whole-server wildcard `--allowedTools mcp__scufris__*` for BOTH
roles. The per-role tool surface is enforced by the SERVER, not the allowlist:
`mcp_server.apply_role` (driven by the `SCUFRIS_AGENT_ROLE` env we set) removes
every tool outside the role before the server serves, so an agent-role server
only ever advertises `request_input`. `--allowedTools` governs auto-approval,
not exposure; a wildcard therefore auto-approves exactly the role's tools and
nothing more.

## Rationale

- Proven live (claude 2.1.193): with `--allowedTools mcp__scufris__*` +
  `--strict-mcp-config` + `--permission-mode default`, claude connected the
  scufris server and CALLED `mcp__scufris__host_stats` unattended
  (`is_error:false`), no approval hang. So the wildcard is accepted; the
  "else enumerate" branch of Step 3 is not needed.
- Faithful parity with codex. Codex approves the whole scufris server
  (`mcp_servers.scufris.default_tools_approval_mode="approve"`), not per tool.
  `mcp__scufris__*` is the claude equivalent of that whole-server approve; an
  enumerated allowlist would make claude MORE restrictive than codex and
  reintroduce the very drift the shared core exists to prevent.
- The backend argv builder stays free of the MCP tool registry: it does not
  have to import `mcp_server` and list tools just to enumerate names. The
  shared core carries only the load-bearing content - the server command/args
  and the role ENV - which is the real single source of truth for scoping.

## Consequences

- The shared core (`agent.scufris_mcp_server`) returns the server's
  `command`, `args`, `env` (no `tool_names` field): the role env is the shared
  truth; the allowlist is a constant `mcp__scufris__*` the claude formatter
  owns.
- `--strict-mcp-config` scopes a scufris-wired claude turn to exactly our
  server, dropping project `.mcp.json` / global config for that turn (the
  spike's "exactly ours" intent). Operator-declared `settings.mcp_servers` are
  NOT wired into claude yet (codex appends them); claude had zero MCP wiring
  before, so this is additive for scufris and a documented follow-up for
  operator servers, not a regression.
