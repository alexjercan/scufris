# Decision: Extract the backend-aware orchestrator diagnostics service

- DATE: 20260803-020105
- STATUS: ACCEPTED
- TASK: 20260729-102148
- TAGS: agents, backend, api

## Context

The scoped `/api/agents/{id}/*` diagnostics resolve backend capability by
comparing a canonical backend NAME at each call site: `_agent_is_codex` in
`scufris/app.py:3252` gates usage/memory/account, `_agent_has_scufris_mcp` at
`scufris/app.py:3360` gates tools/mcp. Two independent name tables, five call
sites, and a fifth backend would have to find all of them.

A base-branch probe (a claude orchestrator via
`PATCH /api/agents/orchestrator`) showed the surface already resolves
backend/model/auth_mode from the record, so the record-drift half of the task
is largely already true. What it does NOT do is distinguish three different
answers:

| Situation | Today's wire | Should be |
|---|---|---|
| codex agent, no rollout yet | `usage: null`, `memory.session_count: 0` | supported, empty |
| claude agent | `usage: null`, `memory.session_count: 0` | unsupported |
| opencode/mock agent | `tools: []` | unsupported |

"No data" and "this backend has no such reader" are the same bytes, so the
panel renders a zero that reads as a measurement.

## Decision

1. **One envelope, not per-capability shapes.** A generic
   `Capability[T]` (`supported: bool`, `value: T | None`) wraps usage, memory,
   the visible-tools listing, and `AccountInfo.quota`. Three states fall out:
   supported+value, supported+empty, unsupported. Verified that pydantic v2
   generics render through FastAPI (`Capability_UsageQuota_` in the OpenAPI
   schema).
2. **`Capability[T]` lives in `scufris/backends/base.py`**, next to
   `BackendStatus`. The diagnostics service imports the backends package, so
   the envelope cannot live in the service without a cycle; `base.py` is the
   seam module and already imports only leaves.
3. **Capability is declared BY the backend, not by a table.** The
   `AgentBackend` protocol gains `read_usage`, `read_memory_footprint`, and a
   `has_scufris_mcp` flag. Codex implements the rollout readers; claude,
   opencode and mock return `Capability` unsupported. This is the seam
   `backends/base.py` already claims to be ("nothing above it branches on which
   backend an agent uses"), so a fifth adapter answers the question by existing.
4. **`AccountInfo.quota` takes the envelope too.** It re-exposes the same usage
   capability; leaving it a bare nullable would keep exactly the silent-empty
   shape this task removes, and would make the rule "envelope, except here".
5. **The scoped wire shape breaks; the frontend is adapted minimally in this
   task.** `web/src/agent-settings-view.ts` and `web/src/agent-types.ts` unwrap
   the envelope so the dashboard is not broken by a landed branch. The richer
   "not supported by this backend" presentation stays with 20260801-100419.

## Alternatives considered

- **A capability table in the service, keyed by canonical backend.** Fewer
  files touched, but it recreates the name comparison one level up - the thing
  the task exists to remove - and a new adapter still fails silently until
  someone remembers the table.
- **Per-capability response models (`UsageReport`, `MemoryReport`, ...).** No
  generics, but four bespoke shapes for one idea, and each new capability
  invents a fifth.
- **Keep the wire shape; add a sibling `supported` field to each model.**
  Non-breaking for the frontend, but it leaves `value` and `supported` free to
  disagree and does not extend to the tools listing (a bare JSON array has
  nowhere to put a flag).
- **Defer the whole frontend change to 20260801-100419.** Rejected: the branch
  would land a dashboard reading `usage.used_percent` off an envelope.

## Consequences

- `/api/agents/{id}/usage`, `/memory`, `/tools` and `AccountInfo.quota` change
  shape. The legacy `/api/agent/*` routes keep theirs; 20260801-100415 moves
  them onto the service and inherits this decision.
- `AgentBackend` grows three members, so every adapter must answer. That is the
  point, and it is the cost: a fifth backend cannot be added without deciding.
- `scufris/app.py` loses the MCP tool helpers (`_tool_parameters`,
  `_as_agent_tool`, `_mcp_servers_for_audience`, `_tools_for_servers`,
  `_probe_servers`) to the service module, which moves it toward the 600-line
  cap it is allowlisted against.
- The diagnostics service must stay under the 600-line source cap; if it does
  not, the split is service vs the MCP tool aggregation, not a new allowlist
  entry.
