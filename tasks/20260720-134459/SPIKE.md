# Spike: operator console (settings page -> debuggable agent status)

- DATE: 20260720-134459
- STATUS: RECOMMENDED
- TAGS: spike, agent, ui

## Question

The read-only Settings page shows config + a bare tool list. What should it become
so a homelab operator can actually answer "why won't the agent do X?" at a glance -
without making it editable yet? Decide the concrete scope for task 122517 (which
of: health checks, env-var names, richer tool cards, versions, session summary,
"try it" runner) so a single flow can build the high-value subset.

## Context

`settings-view.ts` + `GET /api/agent/config` render three cards: Agent (status /
backend / model / auth / sandbox / tools), MCP servers (id + badge), Tools (name +
description). Everything is env-var config, read-only. The round-2 review found the
tool cards "almost useless" (no args, no server, no health) and flagged the biggest
gap: nothing tells the operator whether codex is logged in or the MCP server is
alive - so the agent can "look configured" and silently fail.

Feasibility probed live (all cheap and available):
- `codex login status` -> exit 0 + "Logged in using ChatGPT" (parseable; non-zero /
  other text => not logged in / needs `codex login`).
- `codex --version` -> "codex-cli 0.142.2"; `importlib.metadata.version("scufris")`
  -> "0.1.0".
- MCP tools carry `inputSchema.properties` (arg names, e.g. tatr_ls -> filter, sort)
  - already returned by the in-process `mcp.list_tools()`.
- `settings.web_dist` is the built-assets dir; `list_sessions` gives count +
  `updated_at` (last active).

## Options considered

- **Health card (the headline).** A new `GET /api/agent/health` runs the checks
  (codex installed + version + login; MCP tools reachable via in-process
  list_tools; web assets present) and returns per-check status + a fix hint; the UI
  renders green/amber/red rows. Pro: directly answers "why won't it work"; all
  signals confirmed feasible. Con: shells out to codex (guard with a short timeout).
- **Env-var names beside values.** The Agent card shows the env var that sets each
  value (SCUFRIS_AGENT_MODEL, ...). Cheapest way to make it actionable. Pro: trivial
  (a static name->env map, prefix SCUFRIS_). Con: must stay in sync with config.py.
- **Richer tool cards.** Show the source server + the arg names from `inputSchema`.
  Pro: cheap (extend the tools endpoint), makes the card useful. Con: none real.
- **Versions + session summary.** scufris + codex version; session count + last
  active. Small, orienting. Pro: cheap, useful for bug-hunting across upgrades.
- **"Try it" tool runner (interactive).** Click a tool -> a form from its schema ->
  run it -> show the result, bypassing a chat turn. High value for debugging a
  single tool. BUT: a new run-a-tool endpoint (a real capability, not read-only),
  arg-form generation, result rendering, and a small security/consent surface
  (running host tools on demand). Materially bigger + it breaks the "read-only"
  framing. Smells like its own task.
- **Do nothing.** The page is honest but shallow; the operator still cannot see the
  one thing that most often breaks (auth / MCP up). Not viable given the explicit ask.

## Recommendation

Ship the read-only, high-value subset as one flow (task 122517), and defer the one
interactive item to its own task.

**In scope for 122517 (this flow):**
1. **Health card** via `GET /api/agent/health`: codex (installed + version + logged
   in), MCP server (tools reachable + count), web assets (dist/index.html present),
   agent (enabled + backend). Each = {status: ok/warn/error, detail, hint}. Shell
   commands timeout-guarded; the whole check is best-effort (never 500s the page).
2. **Env-var names** on the Agent config rows (static map; SCUFRIS_ prefix).
3. **Versions**: scufris + codex, shown in the health/about area.
4. **Richer tool cards**: source server + arg names (extend `AgentTool` with `args`
   from `inputSchema.properties`; the tool card lists them).
5. **Session summary**: count + last-active (reuse `/api/agent/sessions` client-side
   or fold a small count into health).

**Deferred to a new task:** the **"try it" tool runner** - interactive, needs a
run-tool endpoint + arg form + consent, and it leaves the read-only framing. Seed
it as its own task; do NOT build it in 122517.

Rationale: the health card is the single biggest debugging win and every other
in-scope item is a cheap, coherent addition to the same page; the runner is a
separate capability with its own risk surface.

## Open questions

- **Health endpoint caching/cost.** `codex login status` + `--version` are
  subprocesses; run them on each health GET with a short timeout, or cache briefly?
  Recommendation: run per-request with a ~2s timeout (the page is opened rarely);
  revisit if it feels slow.
- **"Try it" consent model** (deferred task): running a host tool from the UI needs
  a confirm + should it be gated by a setting? Resolve in that task.

## Next steps

- tatr 20260720-134545: "try it" interactive tool runner (deferred; own risk surface)
- tatr 20260720-122517: build the operator console (health card, env-var names,
  versions, richer tool cards, session summary) - scope decided above; flow now.

## Fix record

- 20260720-122517 (operator console) - LANDED. `scufris/health.py` +
  `GET /api/agent/health` (versions, session summary, agent/codex/auth/mcp/web
  checks with fix hints); Health card + env-var chips + richer tool cards
  (server/args). 135 pytest + 95 frontend green; verified live. "try it" runner
  deferred to 20260720-134545. See tasks/20260720-122517/TASK.md.
