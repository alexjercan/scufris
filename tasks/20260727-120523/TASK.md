# Move MCP per-server health into the Health section; dropdown organizational only

- PRIORITY: 57
- TAGS: feature, mcp, frontend, backend, ui, agents
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Outcome

MCP health moved to the top-of-page Health card as one audience-aware row per
server (orchestrator: `mcp: scufris` + `mcp: den`; sub-agent: `mcp: agent`; no
scufris MCP: a single "none" row); the "MCP tools" dropdown is now purely
organizational (no status dots/bulbs). As-built in `NOTES.md`. Gate green:
`nix flake check` + web `npm run ci` (188 tests + build). DoD tests
`test_agent_health_mcp_rows_are_per_server_and_audience_aware` and
`test_agent_health_den_row_warns_when_unconfigured` pass.

## Story

Follow-up to 20260727-105609 (operator feedback). The green/amber/red MCP
health belongs in the existing top-of-page **Health** card, audience-aware: the
orchestrator's Health shows its MCP servers' tool counts + status, a sub-agent's
Health shows its own callback server's count + status. The "MCP tools" dropdown
stays but becomes purely organizational - drop its summary dot, per-server status
dots, and per-tool bulbs.

Confirmed with the operator: (1) ONE Health row per server (not a single
aggregate), so a degraded server is individually visible (e.g. den amber when
the-den is unconfigured); (2) drop the per-tool bulb entirely - all health
circles live in the Health section; a disabled tool still shows dimmed + its
enable toggle off.

## Steps

- [x] Backend: make the MCP health check in `scufris/health.py` `agent_health`
      AUDIENCE-AWARE and PER-SERVER. Add an `is_orchestrator: bool` parameter;
      replace the single generic "mcp tools" check (currently `_mcp_tool_count`
      summing scufris+den) with one `HealthCheck` per server from
      `mcp_health.servers_for_audience(is_orchestrator)` probed via
      `mcp_health.probe_server`: name `mcp: <id>`, status from the probe
      (ok/warn/error), detail like `N tools` (or the probe's degraded detail,
      e.g. den unconfigured). A backend with no scufris MCP wiring
      (opencode/mock) gets a single `warn` "mcp tools: none (this backend
      exposes no scufris tools)" row. Retire `_mcp_tool_count` if now unused.
- [x] Backend: thread the audience through the two health endpoints in
      `scufris/app.py` - `/api/agent/health` -> `is_orchestrator=True`;
      `/api/agents/{id}/health` -> `is_orchestrator = agent.id ==
      ORCHESTRATOR_ID` and skip the scufris/den probe (single "no scufris tools"
      row) for a backend where `_agent_has_scufris_mcp` is false. Call
      `_ensure_den_path(settings)` before `agent_health` so the in-process den
      probe sees the configured den (mirrors `/api/agent/mcp`).
- [x] Frontend: strip the health visuals from `renderMcpServers`
      (`web/src/settings-view.ts`) - remove the summary dot, the per-server
      status dot in each `<details>` summary, and the per-tool bulb
      (`toolBulb`/`summaryStatus` helpers). Keep the collapsible per-server
      grouping, the tool cards, the enable toggle, the "try it" runner, and the
      existing `tool-card--disabled` dimming. The section is now purely
      organizational.
- [x] Frontend: the Health card (`renderHealthCard`) already renders
      `health.checks`, so the new per-server MCP rows appear there automatically
      - confirm no change needed beyond the backend emitting them. Remove any now
      -dead CSS (`.mcp__summary`, `.tool-card__bulb`, and the `.settings__title-row`
      summary-dot styling) if unused.
- [x] Tests (backend): `tests/test_app.py` / `tests/test_health.py` - the
      orchestrator Health has `mcp: scufris` + `mcp: den` rows with tool counts;
      den is `warn` when unconfigured; a sub-agent (codex/claude) Health has a
      single `mcp: agent` row (2 tools); a mock sub-agent has the "no scufris
      tools" row. Update/replace the old single-"mcp tools"-check assertions.
- [x] Tests (frontend): `web/src/settings-view.test.ts` - `renderMcpServers` no
      longer renders `.health__dot`/`.tool-card__bulb`/`.mcp__summary`; it still
      groups per server and keeps toggles/runners. Drop the bulb/summary
      assertions; keep the grouping + toggle round-trip tests.
- [x] Docs: CHANGELOG note that MCP health moved to the Health card (per-server,
      audience-aware) and the dropdown is organizational; write `NOTES.md`.

## Definition of Done

- The orchestrator settings Health card shows one row per MCP server with its
  tool count and status (`mcp: scufris` green, `mcp: den` green/amber), and a
  sub-agent's Health card shows only its own server's row - audience-aware
  (test: `test_agent_health_mcp_rows_are_per_server_and_audience_aware`).
- den's Health row is amber (`warn`) when the-den is unconfigured
  (test: `test_agent_health_den_row_warns_when_unconfigured`).
- The "MCP tools" dropdown renders NO status dots or per-tool bulbs, only the
  grouped tool cards + controls - the removed health-visual symbols are gone
  (cmd: `grep -n "tool-card__bulb\|mcp__summary\|summaryStatus\|toolBulb" web/src/settings-view.ts`
  returns nothing; `health__dot` legitimately remains in `healthRow`, the Health
  card renderer). Frontend test: `renderMcpServers` renders no `.health__dot`.
- Full gate green: `nix flake check` and `cd web && npm run ci`.
- manual: on the running dashboard, the Health card shows the per-server MCP
  rows (orchestrator two, sub-agent one) and the MCP tools dropdown has no
  coloured circles, only the organized tool cards.

## Notes

- Relevant files: `scufris/health.py` (`agent_health` 91-, the mcp check
  ~205-233, `_mcp_tool_count` 81), `scufris/app.py` (health endpoints 1720-1726,
  1976-1979; `_ensure_den_path`; `ORCHESTRATOR_ID`; `_agent_has_scufris_mcp`),
  `scufris/mcp_health.py` (`servers_for_audience`, `probe_server` - reuse as-is),
  `web/src/settings-view.ts` (`renderMcpServers`, `mcpServerBlock`, `mcpToolCard`,
  `toolBulb`, `summaryStatus`), `web/src/style.css` (`.mcp-server*`,
  `.tool-card__bulb`, `.mcp__summary`, `.settings__title-row`).
- The `/api/agent/mcp` + `/api/agents/{id}/mcp` endpoints stay (the dropdown
  still groups by server from them); the frontend just ignores their status now.
  `McpServerHealth.status`/`AgentTool.available` remain on the models (harmless;
  still the source of the Health rows' status via the parallel backend probe).
- Depends on: 20260727-105609 (CLOSED, landed 6fc2863).
