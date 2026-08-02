# Build Scufris MCP server: curated agent tools (tatr_*, host_stats)

- PRIORITY: 12
- TAGS: feature, backlog, agent, tools, security
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Build the Scufris MCP server exposing a curated, allowlisted set of tools
(`tatr_*`, `host_stats`) to the agent, backed by safe subprocess handlers,
registered with Codex under `[mcp_servers.scufris]`.

## Decisions (from exploration, 20260719)

- Registration touches NO `~/.config`: `codex exec -c 'mcp_servers.scufris.command=...'`
  `-c 'mcp_servers.scufris.args=[...]'` injects the server per-invocation
  (verified: `codex mcp list -c ...` shows it "enabled", nothing written to
  `~/.codex/config.toml`). The agent adds these overrides to each `codex exec`.
- `tatr` is on PATH inside `nix develop` (`~/.nix-profile/bin/tatr` 0.2.0), so the
  MCP server can shell out to it; codex inherits the env we pass.
- MCP lib: use the `mcp` SDK's `FastMCP` (decorator tools, stdio) IF `uv add mcp`
  keeps the venv + flake check green; FALL BACK to a stdlib JSON-RPC stdio server
  if it breaks the build (as openai-codex did). Decide at the first step.

## Steps

- [x] Add `mcp` (`uv add mcp`); verify the venv builds and checks pass. If it
      breaks the uv2nix build, hand-roll a stdlib stdio MCP server instead.
- [x] `scufris/mcp_server.py`: a stdio MCP server exposing read-only tools -
      `host_stats()` (reuse `PsutilCollector` -> `HostStats`), `tatr_ls(filter?)`
      (`tatr ls [-f ...]`), `tatr_show(task_id)` (`tatr show <id>`). A shared safe
      runner: `subprocess.run([...], shell=False, timeout=..., capture)` with
      bounded output; validate args; allowlist == the handlers, nothing generic.
- [x] CLI: `scufris mcp-server` runs the stdio server (so codex spawns it).
- [x] Agent wiring: `_run_codex_exec` injects the `-c mcp_servers.scufris.*`
      overrides (command = `sys.executable`, args = `["-m","scufris.mcp_server"]`)
      when tools are enabled (`agent_tools_enabled`, default on); configure
      whatever approval/tool policy makes MCP calls run non-interactively in
      `codex exec` (determined by live test).
- [x] Tests: each tool handler in isolation (host_stats returns the model;
      tatr_ls/tatr_show against a temp `tasks/` dir with real `tatr`, or a faked
      runner); the safe-runner rejects/handles errors; tool list registered.
- [x] LIVE VERIFY on this host: via `/api/chat`, ask "how much memory is used?"
      (agent calls `host_stats`) and "list my tatr tasks" (agent calls
      `tatr_ls`); confirm the answer reflects real data. Record evidence.
- [x] `ruff`/`mypy`/`pytest` green; app + agent still work with tools off.

## Definition of Done

- The agent can call curated read-only tools (`host_stats`, `tatr_ls`,
  `tatr_show`) through chat, live-verified answering from real host/tatr data.
- Registration is per-invocation via `-c` (no `~/.codex` edits); execution is
  safe (allowlist, `shell=False`, validated args, timeouts, bounded output).
- Tests green with the tool handlers exercised directly; the agent still works
  with tools disabled.

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

## Implementation

- `scufris/mcp_server.py`: a FastMCP (`mcp` SDK) stdio server exposing three
  read-only tools - `host_stats()` (reuses `PsutilCollector`), `tatr_ls(filter?)`,
  `tatr_show(task_id)`. A shared `_run()` helper runs each command with
  `subprocess.run([...], shell=False)`, resolves the exe on PATH, times out, and
  bounds output; the allowlist IS the three handlers (no generic exec).
- `scufris mcp-server` CLI subcommand runs the server; `agent_tools_enabled`
  setting (default on).
- Agent wiring (`_mcp_overrides` in `scufris/agent.py`): every `codex exec` turn
  injects `-c mcp_servers.scufris.command/args` (this interpreter, `-m
  scufris.mcp_server`) plus the approval config below. Nothing is written to
  `~/.codex` (verified: `codex mcp list -c ...` shows it, config.toml untouched).
- Approval (found empirically): unattended `codex exec` auto-cancels MCP tool
  calls ("user cancelled MCP tool call") because there is no stdin to approve on.
  Fixed WITHOUT weakening the sandbox: `-c mcp_servers.scufris.default_tools_approval_mode="approve"`
  + `-c approval_policy="never"`, keeping `--sandbox read-only` as the guardrail.
  (`--dangerously-bypass-approvals-and-sandbox` was rejected - it drops the
  sandbox.)
- Tests (`tests/test_mcp_server.py`): host_stats snapshot, `_run` missing-binary /
  stdout / nonzero-exit, the three tools registered, and `tatr_ls`/`tatr_show`
  against a real `tatr` in a temp tasks dir. ruff+mypy+pytest green.

### Live verification (DoD)

Via `/api/chat` (agent enabled) on this host: "memory used percentage?" ->
`36.6%` (agent called `host_stats`); "how many tatr tasks?" -> `12` (agent called
`tatr_ls`; exactly the 12 task dirs on the branch). Real GPT-5.5 driving real
curated tools, end to end. `codex mcp list -c ...` confirmed the server registers
with no `~/.codex` write.
