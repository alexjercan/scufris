# Spike: give claude-backed agents scufris MCP tools (request_input parity)

- STATUS: OPEN
- PRIORITY: 30
- TAGS: spike,agent,backend,mcp

## Question

How do we wire the built-in scufris MCP server into the CLAUDE backend so a
claude sub-agent can call `request_input` (and, for the orchestrator, the control
tools) - closing the codex-first parity gap - and what exactly does Claude Code's
`--mcp-config` / tool-approval surface require? "Figure out a way" is still open,
and the format details are empirical, so this is a spike (live probe) before code.

## Context (grounded)

- Codex wires MCP per-turn via `agent._mcp_overrides` (`agent.py:153-215`, called
  at `agent.py:366` in `_stream_app_server`): `-c mcp_servers.scufris.*` overrides
  registering `python -m scufris.mcp_server`, role-selected by env
  (`SCUFRIS_AGENT_ROLE`, `SCUFRIS_AGENT_ID`, `SCUFRIS_API_BASE`), with
  `approval_policy="never"` + `default_tools_approval_mode="approve"` so an
  unattended run never hangs on approval.
- The CLAUDE backend (`scufris/backends.py`, `ClaudeBackend` ~356-517;
  `_claude_stream_args` ~357-382) builds `claude -p ... --output-format stream-json
  --verbose --permission-mode <mode> [--resume <id>]` and NEVER adds `--mcp-config`.
  Its `stream` signature already threads `is_orchestrator` / `agent_id` (~471-481)
  but does not use them. Permission mapping (`backends.py:84-88`): manual->default,
  edit->acceptEdits, auto->bypassPermissions (file-write scope, NOT MCP approval).
- The scufris MCP server is backend-agnostic already: `python -m scufris.mcp_server`
  + role env; `apply_role` enforces `_AGENT_ROLE_TOOLS={"request_input"}` for agents
  (`mcp_server.py:644-674`). Only the codex-specific FORMATTING (`-c` flags) is
  codex-bound.
- Prior art / notes: `tasks/20260720-221748/SPIKE.md:169` (names the
  `--mcp-config` / `--allowedTools` gap), `tasks/20260720-223938/NOTES.md`,
  `tasks/20260723-094303` (BC2, explicitly codex-first, tracks this follow-up).

## Unknowns to resolve by LIVE PROBE (real `claude` CLI)

- Exact `--mcp-config` JSON schema (stdio server: command/args/env), and whether
  env can be inlined or must be set on the subprocess.
- Auto-approval for an unattended run: is it `--allowedTools
  "mcp__scufris__request_input"`, `--dangerously-skip-permissions`, or a
  `--permission-mode` value? Confirm it does NOT hang waiting for approval.
- Whether `--permission-mode bypassPermissions` already covers MCP tool calls.
- The tool-name prefix claude uses (`mcp__<server>__<tool>`) for allowlisting.

## Steps

- [ ] Probe `claude --help` (and `claude mcp --help` if present) for the
      `--mcp-config` / `--allowedTools` / permission surface; capture the real flag
      shapes into NOTES.md (close-stdin / background per the codex-probe lessons).
- [ ] Live round-trip: run a real `claude -p` turn with `--mcp-config` registering
      `python -m scufris.mcp_server` (agent role, a test agent id, a local API base)
      and confirm claude can CALL `request_input` unattended (no approval hang), and
      the POST lands a WAITING outcome. This is the empirical proof.
- [ ] Decide the refactor shape: extract a backend-agnostic
      `mcp_server_config(role, agent_id, api_base) -> {command, args, env, tools}`
      core that both codex (`-c`) and claude (`--mcp-config` JSON) format, vs. a
      standalone `_claude_mcp_config` copy. Record the call in a DECISION.md.
- [ ] Write SPIKE.md (question, probe results with real flag shapes, the decided
      approach, risks) and SEED the implementation task(s): wire `--mcp-config` into
      `_claude_stream_args` + tests (argv construction, env threading, disabled-tools
      passthrough), gated behind the live-probe findings.

## Definition of Done

- SPIKE.md records the real Claude Code MCP flags (probed live, not assumed) and a
  decided approach. (manual: SPIKE.md exists with probe output)
- A live claude turn demonstrably calls `request_input` unattended, or the spike
  documents precisely why not and the blocker. (manual: live probe recorded)
- Implementation task(s) seeded in tatr with Steps + DoD (argv + tests), in
  dependency order. (cmd: `tatr ls` shows the seeded task)
- The refactor-vs-copy decision recorded in DECISION.md.

## Notes

- Needs a real `claude` login (live, operator-run) - carries a manual probe, batched
  to the flow Finish if the operator runs it, else the spike records the flag shapes
  from `--help` and seeds impl with a live-probe DoD.
- Feeds the tools panel task (20260723-193216): a claude sub-agent with
  `request_input` then shows it via the backend-aware `tools_for_role`.
- Umbrella 20260723-192825.
