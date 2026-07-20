# Settings page: turn it into an operator console (env names, health, richer tools)

- STATUS: OPEN
- PRIORITY: 30
- TAGS: feature,agent,ui

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
