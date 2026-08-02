# Decision: Delegate legacy /api/agent/* routes to orchestrator diagnostics

- DATE: 20260803-100000
- STATUS: ACCEPTED
- TASK: 20260801-100415
- TAGS: agents, backend, api, frontend

## Context

`20260729-102148` landed `AgentDiagnostics` and the `Capability[T]` envelope,
and put every `/api/agents/{id}/*` panel behind it. Its DECISION.md left the
legacy singular family for this task, saying only that it "inherits this
decision".

The legacy family is not uniform. Three groups:

| Route | Today | Leak? |
|---|---|---|
| `/api/agent/usage`, `/memory`, `/account` | `read_usage(resolve_codex_home(settings))` | yes, unconditional codex |
| `/api/agent/info`, `/api/agent/config` | `settings.agent_model` | yes, codex-only model key |
| `/api/agent/health` | `agent_health(settings, is_orchestrator=True)` | partly: backend follows settings, `has_scufris_mcp` does not |
| `/api/agent/tools`, `/api/agent/mcp` | `*_for_servers(settings, mcp_servers_for_audience(ORCHESTRATOR_ID))` | no |
| app.py Telegram provider `usage()`/`health()` | `read_usage(resolve_codex_home(settings))` | yes |

`settings.agent_model` is the CODEX model slot: `PATCH /api/agents/orchestrator`
writes a model change to `claude_model`/`opencode_model` per backend
(`scufris/app.py:2200`), and the orchestrator record already resolves through
`default_model_for` (`scufris/config.py:443`). So the singular routes report a
codex model for a claude orchestrator while the scoped routes report the right
one.

## Decision

1. **The legacy account family takes the `Capability[T]` envelope.**
   `/api/agent/usage` becomes `Capability[UsageQuota]` and `/api/agent/memory`
   becomes `Capability[MemoryFootprint]`, matching their scoped equivalents.
   `/api/agent/account` already carries the envelope inside `AccountInfo.quota`.
2. **The orchestrator record, not settings, is the source.** Every delegating
   route resolves `_require_agent(ORCHESTRATOR_ID)` and passes the record to the
   service, so a backend switch moves the whole surface at once - the property
   the service exists to provide.
3. **`/api/agent/tools` and `/api/agent/mcp` keep console semantics.** They
   describe the operator console's OWN in-process tool runner
   (`POST /api/agent/tools/{name}/run`), which does not go through the
   orchestrator's backend at all. They already call the shared service helpers
   and read no Codex account.
4. **The `agent_enabled` short-circuit goes.** The scoped routes never had it;
   keeping it would make legacy and scoped disagree for a disabled agent, which
   is the one property the DoD test asserts.
5. **`/api/agent/config.model` is fixed in this task.** It is the same
   `settings.agent_model` lie, one line away from the `/api/agent/info` fix, and
   it feeds the agent-settings page this task is meant to make truthful.

## Alternatives considered

- **Keep the legacy shapes exactly (`UsageQuota | None`, bare
  `MemoryFootprint`) and map `unsupported` to `null` / zeros.** This is what the
  task description asked for, and it is cheaper: no OpenAPI drift, no frontend
  change. Rejected because it reintroduces, on the surface the LANDING PAGE
  reads, precisely the collapse the epic exists to remove: a claude orchestrator
  would report `session_count: 0` - a zero that reads as a measurement. The
  cost of avoiding it is one frontend call site
  (`web/src/agent-view.ts:146`); `/api/agent/memory` and `/api/agent/account`
  have no frontend consumer at all.
- **Make `/api/agent/tools` and `/api/agent/mcp` mirror the scoped routes.**
  Listed in the task's Steps. Rejected: for an opencode or mock orchestrator the
  scoped route reports `supported: false`, so the console's tool list and MCP
  health section would go empty while the in-process runner behind them still
  works. That is a regression invented to satisfy a symmetry the two surfaces
  never had - `/api/agent/tools`' docstring already states the distinction.
- **Delete the legacy family and redirect to the scoped routes.** Out of scope:
  the Notes explicitly keep the compatibility routes, and the landing page,
  Telegram and the release test all still consume them.
- **Wrap the shape change into 20260801-100419 (Telegram + UI).** Rejected for
  the same reason the dependency rejected deferring its own frontend change: the
  branch would land a dashboard reading `usage.primary` off an envelope.

## Consequences

- `/api/agent/usage` and `/api/agent/memory` change wire shape. Consumers: the
  landing page (`loadUsage`, adapted here) and `tests/test_app.py:1816-1905`
  (updated). No other caller.
- A disabled orchestrator now reports its backend's real reading instead of a
  hardcoded null/zero.
- `scufris/app.py` stops importing `read_usage`, `read_memory_footprint` and
  `resolve_codex_home`; the codex account readers are reachable only through
  `scufris/backends/`.
- 20260801-100419 inherits an envelope-shaped `SettingsOps.usage()` boundary:
  this task unwraps at the app.py provider to keep that signature, and 100419
  moves the envelope into the Telegram renderer.
