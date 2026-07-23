# Retro: wire scufris MCP into the claude backend (20260723-201851)

- DATE: 20260723
- OUTCOME: CLOSED, APPROVE (1 review round, 0 blocking findings), landed to master.

## What changed and why

Claude-backed agents now get the built-in role-scoped scufris MCP server wired into
every turn, reaching codex parity: a claude sub-agent can call `request_input` (and
the orchestrator its control tools) unattended.

- Extracted a backend-agnostic core `agent.scufris_mcp_server -> ScufrisMcpServer`
  (a frozen dataclass of `command`, `args`, `env`) from codex's `_mcp_overrides`.
  The role-scoped ENV (`SCUFRIS_AGENT_ROLE` / `SCUFRIS_AGENT_ID` /
  `SCUFRIS_DISABLED_TOOLS`) is the load-bearing content both backends share; each
  formats it to its own flavour (codex `-c` overrides, claude `--mcp-config` JSON).
  This is the same "extract the shared core so the two never drift" move that T2
  used for `role_tool_names`.
- Claude formatter `backends._scufris_claude_args`: inline `--mcp-config` JSON +
  `--strict-mcp-config` + `--allowedTools mcp__scufris__*`, threaded through
  `_claude_stream_args` and `ClaudeBackend.stream`.
- `_agent_has_scufris_mcp` extended to codex OR claude, so a claude sub-agent's
  tools panel shows `request_input` instead of empty.

### Load-bearing decision (DECISION.md)

Whole-server `mcp__scufris__*` allowlist wildcard rather than enumerating the role's
~17 tool names. Confirmed live that claude 2.1.193 accepts the wildcard and runs the
tool unattended. It is role-SAFE because the SERVER enforces the role scope
(`apply_role` reads `SCUFRIS_AGENT_ROLE` and removes out-of-role tools before
serving); the allowlist only governs auto-approval. This also mirrors codex's
whole-server `approval_mode="approve"` posture, so the two backends stay consistent,
and it keeps the argv builder free of the MCP tool registry.

## Difficulties / bugs diagnosed

- The first live wildcard probe FAILED with the scufris server stuck "pending":
  I had launched it with the system python (`/run/current-system/sw/bin/python`),
  which has no `mcp` package. The real interpreter is the nix devshell's
  `python3.14` (what `sys.executable` resolves to inside the running app). Re-ran
  the probe under `nix develop` and it passed. Lesson: probe the scufris MCP server
  with the SAME interpreter the app uses (`sys.executable` in-shell), not whatever
  `python` is first on PATH.
- An early probe's output looked like it leaked my own Claude Code session's tools
  (Bash, ToolSearch, rust-analyzer-lsp) - a red herring from grepping on commas that
  split tool-name JSON. The clean re-run showed the real, single
  `mcp__scufris__host_stats` call.

## What went well

- The spike (20260723-193218) had already proved the mechanism and flagged the
  variadic-`--mcp-config` gotcha, so the impl was aimed, not exploratory. The one
  genuinely-open question (wildcard vs enumerate) was a cheap live probe that
  simplified the design.
- Test-first argv construction pinned the variadic-flag boundary
  (`args[j+2].startswith("--")`) in three orderings including resume.
- The full live loop (real app + real claude subprocess + real HTTP callback)
  reached a durable WAITING outcome, so the DoD's "manual/live" item is genuinely
  proven, not just the argv.

## What to do differently next time

- When live-probing a subprocess-launched MCP server, resolve and use the app's
  interpreter FIRST (one `nix develop --command python -c 'import sys;print(...)'`)
  before building any probe config - would have saved the first failed probe.
- When grepping streamed JSON for tool names, match on the field
  (`grep -o '"name":"[^"]*"'`) rather than splitting on commas.

## Follow-up seeded

- Operator-declared `settings.mcp_servers` are NOT yet wired into claude (codex
  appends them); `--strict-mcp-config` scopes a claude turn to exactly the scufris
  server. Additive (claude had zero MCP before), but operator-server parity for
  claude is a natural next task.
