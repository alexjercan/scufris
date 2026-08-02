# Wire scufris MCP (--mcp-config + allowedTools) into the claude backend

- PRIORITY: 28
- TAGS: feature, agent, backend, mcp
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As a claude sub-agent, I want the scufris MCP server wired into my turn so I can
call `request_input` (and, as the orchestrator, the control tools), reaching codex
parity - so the bidirectional-comms loop and the role-scoped tools view work on
claude too, not just codex.

## Context (grounded, live-probed)

Spike `tasks/20260723-193218/SPIKE.md` proved the mechanism live (claude 2.1.193):
`--mcp-config '<{"mcpServers":{"scufris":{command,args,env}}}>'` +
`--strict-mcp-config` + `--allowedTools "mcp__scufris__request_input"` +
`--permission-mode default` makes claude EXPOSE AND CALL
`mcp__scufris__request_input` unattended. Gotcha: `--mcp-config` is variadic/greedy
- bound it with a following flag, never a positional.

Today `_claude_stream_args` (`scufris/backends.py:357-382`) builds the claude argv
and NEVER adds `--mcp-config`; its `stream` threads `is_orchestrator`/`agent_id`
(unused). Codex wiring lives in `agent._mcp_overrides` (`agent.py:153-215`), the
source of truth for role env + scoping (`role_tool_names`, `_AGENT_ROLE_TOOLS`).

## Steps

- [x] Extract a backend-agnostic core from `_mcp_overrides`:
      `scufris_mcp_server(settings, *, is_orchestrator, agent_id) -> ScufrisMcpServer
      | None` (a frozen dataclass of command/args/env - the role env is the shared
      truth; no tool_names field, superseded by the wildcard, see DECISION.md).
      codex's `_mcp_overrides` now formats it to `-c` overrides, behaviour unchanged
      (guarded by the existing codex tests, all green).
- [x] Add claude formatting in `backends.py` (`_scufris_claude_args`): builds the
      inline `{"mcpServers":{"scufris":{...}}}` JSON and appends `--mcp-config <json>
      --strict-mcp-config --allowedTools mcp__scufris__*`, bounded by the following
      `--strict-mcp-config` flag so the variadic `--mcp-config` cannot eat later args.
      (`--permission-mode` was already on the argv; not duplicated.) Threads
      `is_orchestrator`/`agent_id` from `stream` into `_claude_stream_args`.
      `SCUFRIS_DISABLED_TOOLS` rides the config env.
- [x] Orchestrator allowlist resolved: whole-server `mcp__scufris__*` wildcard
      CONFIRMED live (claude 2.1.193 called `mcp__scufris__host_stats` unattended,
      is_error:false). No enumeration needed; the server enforces role scope. See
      DECISION.md.
- [x] `--resume` still loads `--mcp-config`: the argv is rebuilt every turn, so the
      scufris flags ride resumed turns too (pinned by
      `test_claude_stream_args_keeps_mcp_config_on_resume`).
- [x] Tests (`tests/test_backends.py` + `tests/test_agent.py`): claude argv has
      `--mcp-config` with the right server/env/role, `--allowedTools mcp__scufris__*`,
      `--strict-mcp-config`; `agent_id` threaded; disabled-tools passthrough; no
      config when disabled/no-role; resume keeps config. Extended
      `test_claude_backend_permission_mode_flags`. Shared-core tests added.
- [x] Docs: CHANGELOG entry for claude MCP parity + corrected the "Codex-first" and
      "claude/opencode/mock" caveats. README was already backend-neutral (no change).
- [x] `_agent_has_scufris_mcp` (`app.py`) now includes claude, so a claude
      sub-agent's `/api/agents/{id}/tools` returns `[request_input]`
      (pinned in `test_agent_tools_endpoint_is_role_scoped`).

## Definition of Done

- The claude backend argv registers the scufris MCP server role-scoped, and a
  claude sub-agent calls `request_input`. (test: claude argv construction;
  manual/live: a claude sub-agent blocks via request_input -> WAITING outcome)
- `_agent_has_scufris_mcp` includes claude -> a claude sub-agent's
  `/api/agents/{id}/tools` returns `[request_input]` (was `[]`). (test)
- `ruff check .`, `mypy`, `python -m pytest` + web tests green. (cmd)

## Notes

- Seeded by spike 20260723-193218. Live proof already in SPIKE.md; the impl's live
  DoD is the full loop on claude (a claude sub-agent self-heals like BC5's codex one).
- Umbrella 20260723-192825.
- LIVE PROOF (full loop, claude 2.1.193): booted the real app on the claude backend,
  created a claude sub-agent, ran it with a blocking goal -> claude CALLED
  `mcp__scufris__request_input(question="Should I merge to master?")`, the POST hit
  `/api/agents/<id>/request_input`, and the agent reached a durable WAITING outcome
  (visible in `/api/agents/pending`). Exactly the BC5 self-heal, now on claude.
- Design decision recorded in DECISION.md: whole-server `mcp__scufris__*` allowlist
  wildcard (role-safe, server enforces scope) over enumerating tool names.
- Deferred follow-up (out of scope): operator-declared `settings.mcp_servers` are NOT
  wired into claude (codex appends them). `--strict-mcp-config` scopes a claude turn
  to exactly the scufris server, dropping project `.mcp.json` too. claude had zero MCP
  wiring before, so this is additive for scufris, not a regression; operator-server
  parity for claude is a natural next task.
