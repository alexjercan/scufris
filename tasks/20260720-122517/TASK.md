# Settings page: turn it into an operator console (env names, health, richer tools)

- PRIORITY: 30
- TAGS: feature, agent, ui
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Implementation

- `scufris/health.py` (new): `agent_health(settings)` -> `AgentHealth` (scufris +
  codex versions, session count/last-active, and `HealthCheck` rows for agent /
  codex cli / codex auth / mcp tools / web assets). Best-effort: shell-outs are
  timeout-guarded (`_run`, 3s) and every probe degrades to a check status rather
  than raising; the mock backend treats a missing codex as a warn, not an error.
- `app.py`: `GET /api/agent/health`; `AgentTool` gains `server` + `args` (from the
  tool's `inputSchema.properties`) and `get_agent_tools` populates them.
- `settings-view.ts`: a Health card (version/session line + status-dot rows with
  fix hints) at the top; env-var name chips on the Agent rows (static SCUFRIS_
  map); richer tool cards (source server + arg names); `startSettings` fetches
  health (best-effort, does not blank the page). Unknown check statuses clamp to a
  safe dot class.
- `style.css`: health rows/dots (green/amber/red), env chip, tool-card head/args.

## Tests / verification

- `tests/test_health.py` (new): checks report ok/warn/error per probe (fake
  codex_bin -> deterministic, no real subprocess); disabled agent/tools -> warn.
- `test_app.py`: `/api/agent/health` endpoint shape; `/api/agent/tools` now returns
  `server` + `args` (tatr_ls -> {filter, sort}).
- `settings-view.test.ts`: health card (dots/versions/hint), env-var names, tool
  server/args, unknown-status clamp, health-absent path. 135 pytest + 95 frontend
  green. E2E: live `/api/agent/health` returns all-green (codex logged in, 6 mcp
  tools) - and correctly flags missing `web/dist` before a build.

## Deferred

- The interactive "try it" tool runner -> own task 20260720-134545 (needs a run-
  tool endpoint + consent; leaves the read-only framing). Editable settings ->
  separate spike.

## Goal

Turn the read-only Settings page from a static status page into a useful operator
console that answers "why won't the agent do X?" - without becoming editable yet.

## Notes

- Spikes: tasks/20260720-122301/SPIKE.md (round-3) and tasks/20260720-134459/SPIKE.md
  (operator-console scope decision - READ THIS; feasibility of every signal probed).

## Scope (decided by tasks/20260720-134459/SPIKE.md)

IN this task (read-only operator console):
1. **Health card** via a new `GET /api/agent/health`: codex (installed + version +
   logged in via `codex login status`), MCP server (in-process `list_tools` count),
   web assets (`settings.web_dist`/index.html present), agent (enabled + backend).
   Each = {status: ok|warn|error, detail, hint}. Shell-outs timeout-guarded; the
   check is best-effort and must NEVER 500 the page.
2. **Env-var names** beside each Agent config row (static SCUFRIS_ map).
3. **Versions**: scufris (`importlib.metadata.version`) + codex (`--version`), in
   the health/about area.
4. **Richer tool cards**: source server + arg names from the tool `inputSchema`
   (extend `AgentTool` with `args: list[str]`).
5. **Session summary**: count + last-active (reuse `/api/agent/sessions` or fold
   into health).

DEFERRED (own task 20260720-134545): the interactive **"try it" tool runner** - it
is interactive, needs a run-tool endpoint + consent, and leaves the read-only
framing.

- Editable settings / switching the model stays DEFERRED (separate spike).
- Frontend + backend health endpoint; escape everything; jsdom-safe (side-effect-
  free render).
