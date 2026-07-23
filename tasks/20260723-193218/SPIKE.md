# Spike: give claude-backed agents scufris MCP tools (request_input parity)

- DATE: 20260723
- STATUS: RECOMMENDED (live-probed; mechanism proven end to end)
- TAGS: spike, agent, backend, mcp

## Question

How do we wire the scufris MCP server into the CLAUDE backend so a claude
sub-agent can call `request_input` (and the orchestrator the control tools),
closing the codex-first parity gap - and what exactly does Claude Code's
`--mcp-config` / tool-approval surface require? Empirical, so probed live.

## Probe results (claude 2.1.193, live)

Real flags (`claude --help`, `claude mcp --help`, `claude mcp add-json --help`):

- `--mcp-config <configs...>` - loads MCP servers from JSON FILES **or an inline
  JSON string**. VARIADIC and GREEDY: it consumes every following token as another
  config path until the next `--flag`. In a probe, `--mcp-config "$JSON" mcp list`
  swallowed `mcp` and `list` as config paths ("config file not found: .../mcp").
  So it MUST be followed by a flag, never by a positional. In `_claude_stream_args`
  we append flags after it (`--allowedTools`, `--strict-mcp-config`, `-p`), so it is
  bounded - but this is a real gotcha to pin with a test.
- `--strict-mcp-config` - use ONLY the servers from `--mcp-config`, ignoring project
  `.mcp.json` / global config. Use it so a sub-agent's surface is exactly ours.
- `--allowedTools <tools...>` - auto-approve list. MCP tools are named
  `mcp__<server>__<tool>`, i.e. `mcp__scufris__request_input`.
- `--permission-mode <mode>` (default/acceptEdits/bypassPermissions) +
  `--dangerously-skip-permissions`. With `--allowedTools` covering the scufris tool,
  `--permission-mode default` runs UNATTENDED (no approval hang).

The JSON schema (Claude Code standard, confirmed accepted):

    {"mcpServers": {"scufris": {"command": "python",
       "args": ["-m", "scufris.mcp_server"],
       "env": {"SCUFRIS_AGENT_ROLE": "agent", "SCUFRIS_AGENT_ID": "<id>",
               "SCUFRIS_API_BASE": "http://127.0.0.1:<port>",
               "SCUFRIS_DISABLED_TOOLS": "..."}}}}

### The load-bearing proof (live turn)

    claude --strict-mcp-config --mcp-config <scufris.json> \
      --allowedTools "mcp__scufris__request_input" --permission-mode default \
      --output-format stream-json --verbose \
      -p "You are blocked... Call the request_input tool with question=..."

-> the stream contained `"name":"mcp__scufris__request_input"` with `tool_use`:
claude EXPOSED and CALLED the scufris `request_input` tool, unattended. Same env
(`SCUFRIS_AGENT_ROLE=agent`, `SCUFRIS_AGENT_ID`) as codex; the server's `apply_role`
scopes it to `request_input` regardless of backend. The mechanism is proven.

## Decision (backend-agnostic core + per-backend formatters)

Extract the "role -> (server command, args, env, tool names)" logic that today
lives inside codex's `agent._mcp_overrides` into a backend-agnostic core, and have
each backend FORMAT it:

- codex: `-c mcp_servers.scufris.*` overrides + `approval_policy="never"` +
  `default_tools_approval_mode="approve"` (unchanged behaviour).
- claude: `--mcp-config '<json>'` (inline, bounded by following flags) +
  `--strict-mcp-config` + `--allowedTools "mcp__scufris__<tool>[ ...]"` (the role's
  tool names, `mcp__scufris__` prefixed) so the unattended turn never hangs.

Rationale: the CONTENT (command, env, role scoping) is identical across backends
and already the source of truth for scoping; only the FORMAT differs. A shared core
keeps codex and claude from drifting on what a role exposes (the same reason T2
extracted `role_tool_names`). See DECISION.md.

## Remaining unknowns for the impl (live-probe DoD)

- Orchestrator `--allowedTools`: the agent role is one tool (`request_input`); the
  orchestrator has ~17. Confirm whether claude accepts a whole-server wildcard
  (`mcp__scufris` / `mcp__scufris__*`) or needs each name enumerated. Enumerate from
  `role_tool_names` if no wildcard.
- Session resume (`--resume`) + `--mcp-config` interaction: confirm a resumed claude
  turn still loads the config (the scufris backend re-sends args each turn, like
  codex re-sends the sandbox - see the codex resume lesson).
- `SCUFRIS_DISABLED_TOOLS` passthrough via the config `env`.

## Seeded task

- IMPL `20260723-201851` (p28): wire `--mcp-config` + `--allowedTools` into the
  claude backend from the shared core, with argv-construction tests and the live
  round-trip as the acceptance DoD.
