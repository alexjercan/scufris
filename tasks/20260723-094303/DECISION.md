# Decision: how sub-agents get the `request_input` tool (role-scoped, one server)

- DATE: 20260723
- TASK: 20260723-094303 (BC2)
- STATUS: ACCEPTED
- SPIKE: `tasks/20260723-001256/SPIKE.md` (Q2)

## Context

BC2 gives sub-agents a `request_input` callback tool. Sub-agents currently get
NO scufris MCP tools: `_mcp_overrides` (`agent.py:153-194`) registers the whole
`scufris` server ONLY when `is_orchestrator` (`app.py:1106`). So BC2 needs a way
to expose exactly one tool to sub-agents while the orchestrator keeps the full
surface. Two shapes were considered.

## Reframing T3 (the input that decided this)

T3 (`tasks/20260722-222729`) was NOT a hard "sub-agents get nothing from scufris"
security boundary. Per the user, it was a CAPABILITY preference: "none of the
current MCP tools are useful for sub-agents, and I don't want sub-agents creating
or running other agents." So the requirement BC2 must preserve is narrow -
sub-agents must not reach the control tools (`create_agent`/`run_agent`/
`message_agent`/project CRUD) - not "physically isolate sub-agents from the
server binary". This removes the weight that a physical boundary would carry.

## Options

### A - separate minimal MCP server (rejected)

A second FastMCP entry point (`scufris-callback`/`scufris-agent`) exposing only
`request_input`, registered for non-orchestrator agents; the existing `scufris`
server stays orchestrator-only.

- Pro: a physical boundary - the sub-agent server cannot expose control tools
  because that code is not in it.
- Con: a second entry point plus duplicated HTTP plumbing (base-URL env,
  `_api_call`, the app-endpoint call path). The physical-boundary pro only
  matters under a hard-isolation requirement, which the T3 reframe says does not
  exist.

### B - one server, role-scoped tools (ACCEPTED)

Keep ONE `scufris` server. Generalize the `is_orchestrator` boolean in
`_mcp_overrides` into a ROLE/audience (`orchestrator` vs `agent`). Tag each tool
with its audience (control/observe/host = `orchestrator`; `request_input` =
`agent`; a tool may be `both`). Always register the server, pass the role via
env, and the server exposes only that audience's tools at startup - an
ALLOWLIST-by-role, reusing the `codex-per-server-env-filters-mcp-tools`
machinery (today a `SCUFRIS_DISABLED_TOOLS` denylist; this becomes a role
allowlist).

- Pro: one codebase, one entry point, shared HTTP plumbing, reuses existing
  gating. The capability guarantee (sub-agents cannot reach control tools) is an
  explicit allowlist in one place - easy to read and test.
- Pro: models the real shape - TWO audiences with different useful toolsets, not
  "orchestrator vs nothing". Adding the next sub-agent-useful tool later (e.g. a
  `report_progress`, or a read-only `host_stats`) is a one-line audience tag, no
  new server.
- Con: the sub-agent's process still LOADS the control-tool code (does not
  advertise it) - a runtime filter, not a physical absence. Acceptable given T3
  is a capability preference, not a sandbox.

## Decision

Adopt **B**: one `scufris` server, role-scoped tool exposure. Promote
`is_orchestrator` to a role/audience model; `request_input` is the first `agent`
audience tool. Frame BC2 as "add a sub-agent-audience tool via a role model that
generalizes the orchestrator gate", NOT as "reverse a T3 security boundary" -
because T3 was a capability preference, and the guarantee it cares about
(no control tools for sub-agents) is preserved by the allowlist.

## Consequences

- The `is_orchestrator` plumbing that BC3/BC4 already assume is generalized, not
  duplicated. BC3's orchestrator-only `pending_agents`/`acknowledge` become
  `orchestrator`-audience tools under the same model.
- The tool-audience allowlist must be explicitly tested: sub-agent role exposes
  ONLY `request_input`; orchestrator role exposes the control/observe/host set.
- Still codex-first: claude sub-agents get no scufris MCP wiring today
  (`backends.py` never adds `--mcp-config`); the claude parity gap is a tracked
  follow-up, unchanged by A-vs-B.
</content>
