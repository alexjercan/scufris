# Review: agent reach - config-driven MCP registry + more Scufris tools

- DATE: 20260719
- VERDICT: APPROVE (1 round)

## Scope reviewed

`scufris/mcp_server.py` (`disk_usage`, `list_processes`, `_format_processes`),
`scufris/config.py` (`McpServerSpec` + `Settings.mcp_servers`), `scufris/agent.py`
(`_mcp_overrides` refactor + `_server_override`), tests
(`test_mcp_server.py`, `test_agent.py`, new `test_config.py`).

## Correctness

- Both new tools live-verified against the real host: `disk_usage()` returned the
  real `df -h` table (root 51%, /boot), `list_processes(5)` returned real top apps
  (rustc, python3.14, .claude-wrapped, firefox) via the pure formatter. `/api/agent/
  tools` now lists `[disk_usage, host_stats, list_processes, tatr_ls, tatr_show]`.
- Security model intact: both tools are read-only, fixed arg lists (no shell),
  and inherit `_run`'s timeout + `_MAX_OUTPUT` cap; `list_processes` uses the
  psutil collector (no subprocess) and is bounded by the collector's top-N.
  `df` excludes tmpfs/devtmpfs/squashfs/overlay noise.
- The registry refactor is backward-compatible: with an empty `mcp_servers`
  (the default), `_mcp_overrides` emits a byte-identical scufris block +
  `approval_policy="never"` - verified in the smoke, so existing codex-exec
  behavior is unchanged. A configured server appends its own block after.
- Injection-safe: a server `id` becomes a TOML key, so it is validated against
  `^[A-Za-z0-9_]+$` and the reserved `scufris` id is skipped - a spec with
  `id="bad.id"` or `id="scufris"` (command "evil") produces neither key (tested).
  `command`/`args` go through `json.dumps` (TOML value escaping) and come from
  operator config, not the model.
- External servers are opt-in (empty default) and the operator owns each binary +
  the `approve` trust decision; `approve=false` omits the auto-approval line
  (tested). This matches the spike's "gated + off by default" requirement.
- The documented entry point works: `SCUFRIS_MCP_SERVERS` JSON parses into
  `McpServerSpec` list (new `test_config.py`), and the programmatic path is
  covered by the `_mcp_overrides` tests.
- `test_tools_registered` updated to the new exact set + non-empty descriptions.
- Full suite green: `ruff`/`ruff format`/`mypy` (10 files)/`pytest`.

## Nits (non-blocking)

- `df` still lists `efivarfs` (a tiny real pseudo-fs); harmless, not worth another
  `-x`.
- `list_processes` CPU% can exceed 100 (per-core sum, e.g. rustc 980%) - correct
  psutil semantics, informative for a build box.
- `_human_bytes` duplicates the frontend's `formatBytes` in Python; there was no
  existing Python helper to reuse, so a small local one is fine.

## Verdict

APPROVE. The agent gained two curated read-only tools and MCP servers are now
config-declared without disturbing the built-in Scufris registration, with id
validation closing TOML-key injection and externals gated off by default.
Live-verified; security model preserved.
