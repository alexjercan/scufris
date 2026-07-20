# Settings page: turn it into an operator console (env names, health, richer tools)

- STATUS: OPEN
- PRIORITY: 30
- TAGS: feature,agent,ui

## Goal

Turn the read-only Settings page from a static status page into a useful operator
console that answers "why won't the agent do X?" - without becoming editable yet.

## Notes

- Spike: tasks/20260720-122301/SPIKE.md. Round-2 review findings.
- Additions (all read-only or lightly interactive):
  - show the env-var NAME beside each value (SCUFRIS_AGENT_MODEL, _BACKEND,
    _TOOLS_ENABLED, ...), consistently (today only the tools-disabled line does).
  - live health checks: is codex logged in (`codex login status`)? is the scufris
    MCP server reachable (list_tools succeeds)? is `web/dist` present? Render
    green/amber/red with the fix hint.
  - version info (scufris + codex --version).
  - richer tool cards: source server, and the arg schema (name/params) from the
    MCP tool definition; optionally a "try it" runner that calls one tool and shows
    the result (bypassing a full chat turn).
  - a small session summary (count, last-session time).
- Editable settings / switching the model stays DEFERRED to its own spike.
- Frontend + a backend health/config endpoint; escape everything; jsdom-safe.
