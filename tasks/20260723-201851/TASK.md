# Wire scufris MCP (--mcp-config + allowedTools) into the claude backend

- STATUS: OPEN
- PRIORITY: 28
- TAGS: feature,agent,backend,mcp

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

- [ ] Extract a backend-agnostic core from `_mcp_overrides`:
      `mcp_server_config(settings, *, is_orchestrator, agent_id) -> {command, args,
      env, tool_names}` (tool_names = the role's scufris tool names, for allowlisting)
      - or a small dataclass. Keep codex's `_mcp_overrides` formatting it to `-c`
      overrides (behaviour unchanged; guard with the existing codex tests).
- [ ] Add claude formatting in `backends.py`: build the `{"mcpServers":{"scufris":
      {...}}}` JSON (inline), and append `--mcp-config <json> --strict-mcp-config
      --allowedTools <mcp__scufris__* names> --permission-mode <mode>` to the argv,
      bounded so the variadic `--mcp-config` cannot eat later args. Thread
      `is_orchestrator`/`agent_id` from `stream` into `_claude_stream_args`.
      Passthrough `SCUFRIS_DISABLED_TOOLS`.
- [ ] Resolve the orchestrator allowlist: confirm (live) whether claude accepts a
      whole-server `mcp__scufris` / `mcp__scufris__*` wildcard; else enumerate the
      role's tool names via `role_tool_names`.
- [ ] Confirm `--resume` still loads `--mcp-config` (args re-sent each turn, like the
      codex sandbox-on-resume lesson); add it if a resumed turn drops the config.
- [ ] Tests (`tests/test_backends.py`): the claude argv contains `--mcp-config` with
      the right server/env/role, `--allowedTools` with the role's `mcp__scufris__*`
      names, `--strict-mcp-config`; `agent_id` threaded; disabled-tools passthrough.
      Extend `test_claude_backend_permission_mode_flags`.
- [ ] Docs: CHANGELOG note claude reaches MCP parity; update the README/CHANGELOG
      "Codex-first (claude sub-agents have no scufris MCP wiring yet)" caveat.
- [ ] Once claude wires MCP, generalize `_agent_has_scufris_mcp` (`app.py`, added by
      task 20260723-193216) to include claude, so a claude sub-agent's tools panel
      shows `request_input` instead of empty.

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
