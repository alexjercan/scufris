# Review: Scufris MCP server (curated agent tools)

## Round 1 - 20260719

Scope: `scufris/mcp_server.py`, `scufris/agent.py` (`_mcp_overrides`),
`scufris/cli.py` (`mcp-server`), `scufris/config.py` (`agent_tools_enabled`),
`tests/test_mcp_server.py`, `mcp` dependency.

### Correctness

- Proven end to end on this host via `/api/chat`: the agent called `host_stats`
  ("36.6%") and `tatr_ls` ("12" - exactly the branch's task count). Real GPT-5.5
  driving real curated tools, not a mock.
- Registration touches NO `~/.config`: per-invocation `-c mcp_servers.scufris.*`;
  verified `codex mcp list -c ...` shows it while `~/.codex/config.toml` stays
  clean. Matches the user's "nix-way, don't touch ~/.config" constraint.
- The unattended-approval trap (MCP calls auto-cancelled with stdin closed) was
  diagnosed from the real error and fixed the RIGHT way: auto-approve only this
  server's tools + `approval_policy="never"`, while KEEPING `--sandbox read-only`.
  The blunt `--dangerously-bypass-approvals-and-sandbox` was declined because it
  drops the sandbox - good call.
- Safe execution: `_run` uses `subprocess.run([...], shell=False)`, resolves the
  exe on PATH, times out, and bounds output; the allowlist is literally the three
  handlers - no generic "run any command" path (matches the spike's security
  stance).
- Tests exercise the real `tatr` against a temp tasks dir and the real collector,
  plus `_run` edge cases and tool registration. ruff+mypy+pytest green; `mcp`
  installs cleanly in uv2nix (no bundled-binary trap like openai-codex).

### Observations (non-blocking)

- MINOR: `tatr_ls`/`tatr_show` run in codex's cwd, so "my tasks" means whatever
  project the backend runs from. Fine for the single-host dashboard; a
  configurable tatr root (`tatr -r`) is a future refinement if wanted.
- MINOR: `approval_policy="never"` also auto-approves the model's shell commands,
  but they remain confined by `--sandbox read-only`, so the guardrail holds. The
  read-only sandbox is the security boundary, not the approval prompt.
- NIT: codex emits unrelated `HTTP 451 no_biscuit`/`503 circuit_open` transport
  noise while reaching some remote service; it does not affect the stdio tool
  calls (which complete). Cosmetic.

### Verdict

- VERDICT: APPROVE

Meets the Definition of Done: the agent runs curated read-only tools
(`host_stats`, `tatr_ls`, `tatr_show`) through chat, live-verified from real
data; registration is `~/.config`-free; execution is allowlisted and sandboxed;
checks are green with real tool tests. MINOR items are single-user-appropriate.
