# Agent reach: config-driven MCP server registry + more Scufris tools

- PRIORITY: 15
- TAGS: feature, agent, mcp, spike
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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

## Decisions (from /plan)

- Two new read-only Scufris tools (both fit the "host introspection" theme, both
  bounded/timeout via the existing `_run`): `disk_usage()` (`df -h`, noise types
  excluded) and `list_processes(limit)` (top application groups via the existing
  `PsutilProcessCollector` + a pure formatter). Stay read-only this cycle - a
  write tool like `tatr_new` changes the server's posture and is out of scope.
- Registry: a `McpServerSpec` (id, command, args, approve) in `config.py` and
  `Settings.mcp_servers: list[McpServerSpec] = []` (empty default -> only the
  built-in `scufris` runs, so nothing changes unless the operator declares more,
  e.g. via `SCUFRIS_MCP_SERVERS` JSON). `_mcp_overrides` becomes: emit the
  built-in `scufris` block (unchanged), then one block per configured server,
  then the global `approval_policy="never"`. Validate each `id` against
  `^[A-Za-z0-9_]+$` and reserve `scufris`, so a spec cannot inject TOML keys.
- External servers are OFF by default and gated by config; the operator supplies
  the binary + accepts the security trade (writes/network vs the read-only
  sandbox). No generic "run any command" tool.

## Steps

- [x] `scufris/mcp_server.py`: add `disk_usage()` (`_run(["df","-h","-x",...])`)
      and `list_processes(limit=15)` (module `PsutilProcessCollector`, a pure
      `_format_processes(plist, limit) -> str`). Keep the read-only + bounded
      contract.
- [x] `scufris/config.py`: `McpServerSpec {id, command, args=[], approve=True}` +
      `Settings.mcp_servers: list[McpServerSpec] = []`.
- [x] `scufris/agent.py`: refactor `_mcp_overrides` to a per-server
      `_server_override(id, command, args, approve)` helper, emit the built-in
      `scufris` server first then each configured spec (skip invalid/reserved id),
      then `approval_policy="never"`. Byte-identical output when `mcp_servers` is
      empty.
- [x] Tests: `test_mcp_server.py` (update `test_tools_registered` to the new set;
      `_format_processes` pure; `disk_usage`/`list_processes` return sane text);
      `test_agent.py` (`_mcp_overrides` default = the scufris block; an extra spec
      adds its block; invalid id skipped; disabled -> []).
- [x] `nix develop` full check green (ruff, mypy, pytest) + a live smoke: the
      agent's tool list (`/api/agent/tools`) includes the new tools, and the two
      tools return real output.

## Definition of Done

- The agent has two new read-only tools (disk usage, process list) and MCP
  servers are config-declared (a list of specs -> `-c` blocks), with the built-in
  `scufris` server unchanged by default and ids validated. Security model intact
  (fixed arg lists, timeouts, bounded output, read-only sandbox, trusted-only
  auto-approve). `ruff`/`mypy`/`pytest` green; live-verified.

## Implementation

- `mcp_server.py`: two read-only tools - `disk_usage()` (`df -h` minus
  tmpfs/devtmpfs/squashfs/overlay) and `list_processes(limit=15)` (module
  `PsutilProcessCollector` + pure `_format_processes` fixed-width table).
- `config.py`: `McpServerSpec {id, command, args, approve}` +
  `Settings.mcp_servers: list[McpServerSpec] = []` (JSON via SCUFRIS_MCP_SERVERS).
- `agent.py`: `_mcp_overrides` refactored to a per-server `_server_override`
  helper - emits the built-in scufris block (byte-identical when no extras) then
  each configured spec, then `approval_policy="never"`. Ids validated against
  `^[A-Za-z0-9_]+$`; the reserved `scufris` id is skipped (no TOML-key injection).
- Tests: updated `test_tools_registered` to the 5-tool set; `_format_processes`
  pure test; `disk_usage`/`list_processes` integration; `_mcp_overrides` default /
  disabled / appended-server / invalid+reserved-id; new `test_config.py` for the
  env JSON. Live: real df + top processes; `/api/agent/tools` lists both new tools;
  a configured server appends cleanly. `ruff`/`mypy`/`pytest` green.

## Notes

- Spike: tasks/20260719-212152/SPIKE.md.
- Preserve the security model: fixed arg lists (never shell strings), timeouts,
  bounded output, read-only sandbox, auto-approve only trusted servers. codex's
  own skills/plugins system is out of scope (codex-managed, not our per-invocation
  injection path).
- This is the last open task from the agent-page expansion spike.
