# Agent reach: config-driven MCP server registry + more Scufris tools

- STATUS: OPEN
- PRIORITY: 15
- TAGS: feature, agent, mcp, spike

## Goal

Give the agent more to work with, on two axes:

1. **Cheapest:** add more curated read-only `@mcp.tool()` handlers to
   `scufris/mcp_server.py` (candidates: log tailing, `df`/disk usage,
   `systemctl --user status`, git status of a repo, more tatr queries). No new
   process - it is already in the nix closure.
2. **Registry:** generalize `_mcp_overrides` from the hard-coded `scufris` block
   into a config-declared LIST of server specs (id, command, args, approval
   mode), each emitted as `-c mcp_servers.<id>.*`. Adding a server becomes config,
   not code.

Feasibility (from the spike): registering more servers via `-c` is exactly how
Scufris is registered today, so YES. The "something else" needed for EXTERNAL
servers is (a) each server's binary on the host (nixpkgs / npx / uvx) and (b) a
security gate - external servers may want writes/network, which fights the
read-only sandbox, so keep them config-declared and OFF by default. No generic
"run any command" tool - curated handlers only.

## Notes

- Spike: tasks/20260719-212152/SPIKE.md.
- Independent of the sessions/context/usage tasks; can flow in parallel.
- Preserve the security model: fixed arg lists (never shell strings), timeouts,
  bounded output, read-only sandbox, auto-approve only trusted servers. codex's
  own skills/plugins system is out of scope (codex-managed, not our per-invocation
  injection path).
