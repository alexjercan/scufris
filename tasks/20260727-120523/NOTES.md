# NOTES - Move MCP per-server health into the Health card

As-built record for 20260727-120523 (operator-feedback follow-up to
20260727-105609). What/why in TASK.md; this is the design/fix log.

## What changed

### Backend
- `scufris/health.py` `agent_health` gained `is_orchestrator: bool = True` and
  `has_scufris_mcp: bool = True`. The single non-audience-aware "mcp tools" check
  (which used `_mcp_tool_count` to sum the orchestrator's in-process scufris+den
  tools - wrong on a sub-agent page) is replaced by ONE `HealthCheck` per server
  for the audience, via `mcp_health.servers_for_audience(is_orchestrator)` +
  `probe_server`: `mcp: scufris` / `mcp: den` for the orchestrator, `mcp: agent`
  for a sub-agent. `agent_tools_enabled=false` -> a single "mcp tools" warn row;
  `has_scufris_mcp=false` (opencode/mock) -> a single "none" warn row.
  `_mcp_tool_count` deleted (unused).
- `scufris/app.py` the two health endpoints thread the audience:
  `/api/agent/health` -> `is_orchestrator=True`; `/api/agents/{id}/health` ->
  `is_orchestrator = agent.id == ORCHESTRATOR_ID` + `has_scufris_mcp =
  _agent_has_scufris_mcp(agent)`. Both call `_ensure_den_path(settings)` first so
  the in-process den readiness check sees the configured den (same bridge the
  `/api/agent/mcp` endpoint uses).

### Frontend
- `web/src/settings-view.ts` `renderMcpServers` is now purely organizational:
  removed the summary status dot, the per-server `<details>` status dot, and the
  per-tool `toolBulb` (and the `summaryStatus`/`toolBulb` helpers). Kept the
  per-server collapsible grouping, tool cards, enable toggle, "try it" runner, and
  the `tool-card--disabled` dimming. The Health card (`renderHealthCard`, which
  already renders `health.checks`) shows the new per-server rows with no change.
- `web/src/style.css`: removed `.settings__title-row`, `.mcp__summary`,
  `.mcp-server__detail`, `.tool-card__bulb` (dead).

## Difficulties / decisions

- The MCP health is now sourced TWICE from the same in-process probe: the Health
  rows come from `agent_health` (backend) and the dropdown grouping from
  `/api/agent/mcp`. Both call `mcp_health.probe_server`, so they cannot disagree;
  the small duplicate probe cost is acceptable (in-process, cheap) and keeps the
  Health card a pure `AgentHealth.checks` surface rather than frontend-composed.
- The DoD grep was initially too broad (`health__dot` legitimately remains in
  `healthRow`, the Health-card renderer); tightened it to the actually-removed
  symbols (`tool-card__bulb`, `mcp__summary`, `summaryStatus`, `toolBulb`).
- Applied the lesson from the previous cycle: formatted ONLY the touched files
  (`ruff format <files>` / `prettier --write <files>`), not whole dirs, so no
  unrelated formatter drift entered the diff.

## Self-reflection

- Small, well-scoped follow-up. The previous cycle's mistakes (nix-flake-check
  untracked-files, ruff format scope) did not recur because their ledger entries
  were fresh. That is the compounding working as intended.
