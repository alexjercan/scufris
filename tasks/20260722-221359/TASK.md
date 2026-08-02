# Spike: Telegram frontend - orchestrator-as-the-whole-UI, MCP control tools, orchestrator-only tool scoping

- PRIORITY: 0
- TAGS: spike, feature, agent, mcp, frontend, telegram, backlog
- KIND: SPIKE
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As the operator of my single NixOS box, I want to drive all of Scufris from a
Telegram chat - the way my old `github.com/alexjercan/scufris-bot` let me talk to
the machine from my phone - so that I do not need the web dashboard open to see
host stats, spin up an agent on a project, or check what an agent is doing.

The key framing: Telegram is a single chat, so there is exactly ONE session, and
that session IS the orchestrator agent. The Telegram frontend is therefore not a
reimplementation of the dashboard; it is a second face on the orchestrator. The
orchestrator already can do most of what the dashboard does by talking to the
backend - what is missing is (a) MCP tools that let it DO the control actions the
dashboard exposes (create an agent for a project, run/steer it, create a project,
edit settings), not just observe, and (b) a Telegram transport that maps the chat
to the orchestrator's session.

This spike decides the feature set and the architecture before any of it is
planned into buildable steps. It is fuzzy on purpose - the real question is "what
should the Telegram face even do, and how do the orchestrator's tools need to
change to support it".

## What the spike must decide

1. **Feature scope.** Which slices of the dashboard are worth exposing through a
   chat-only surface, and which are not? Candidates: live host metrics, agent
   list/status/create/run/steer, project list/create/detail, tatr backlog,
   settings. Decide the v1 cut vs later.

2. **MCP control-tool set.** Enumerate the `scufris/app.py` HTTP endpoints (it is
   the full control surface, ~1700 lines) and decide which need a curated MCP
   tool wrapper so the orchestrator can ACT. Today `mcp_server.py` has 8 tools
   and the agent-facing ones (`list_agents`, `agent_status`) are READ-ONLY. We
   need write/control tools: e.g. `create_agent(project, backend, ...)`,
   `run_agent`/`steer_agent`, `create_project`, `list_projects`. Name the set and
   map each tool to its endpoint, keeping the `_run`/curation contract (fixed
   args, bounded output, no arbitrary shell).

3. **Orchestrator-only tool scoping.** This is the crux. Today
   `scufris/agent.py:_mcp_overrides(settings)` registers the scufris MCP for
   EVERY agent off GLOBAL settings (`agent_tools_enabled`, `disabled_tools`) -
   there is no orchestrator-vs-agent distinction. The new control tools must be
   available ONLY to the orchestrator; regular agents keep getting their tools
   from their own `.config` / project `.skills` and should NOT be able to create
   agents or projects. Decide the mechanism: an `is_orchestrator` flag threaded
   into `_mcp_overrides`, a separate tool subset, or a `SCUFRIS_ORCHESTRATOR` env
   gate the MCP server reads (mirroring today's `SCUFRIS_DISABLED_TOOLS`).

4. **Which existing MCP tools to keep, drop, or relocate.** `tatr` is a skill the
   agent can already run via `Bash`, so `tatr_ls`/`tatr_show`/`tatr_new` may not
   need to exist as MCP tools at all (discuss: keep for non-orchestrator agents
   that lack Bash, or drop entirely). Same question for
   `host_stats`/`disk_usage`/`list_processes` - are these orchestrator-only host
   introspection now? Decide the minimal curated set per audience.

5. **Telegram transport + UX.** Long-poll vs webhook; how a Telegram chat maps to
   the single orchestrator session (persist the session id, `/new` to reset);
   auth (allowlist of chat ids - the box must not answer strangers); token via
   pydantic-settings/`.env`; streaming vs final-only replies and how (or whether)
   tool-call progress renders in Telegram; run alongside the FastAPI app or as a
   separate process/entry point. Reference the old `scufris-bot` for what worked.

## Candidate decomposition (the spike confirms and seeds these)

- **T1 - Orchestrator-only scoping.** Thread orchestrator identity into
  `_mcp_overrides` + `mcp_server.py` so a control-tool subset is gated to the
  orchestrator only. (This unblocks everything else.)
- **T2 - Control MCP tools** over the chosen app endpoints (create/run/steer
  agent, create/list project, ...), curated and bounded, with tests.
- **T3 - Prune/relocate existing tools** per the decision (drop or keep
  `tatr_*`, scope host tools).
- **T4 - Telegram transport module**: long-poll/webhook, chat->orchestrator
  session mapping, auth allowlist, token config, entry point.
- **T5 - Reply rendering + proof**: stream vs final, tool chips, plus an
  `examples/` script and an integration test against a stubbed Telegram API and
  stubbed backend.

## Definition of Done

- `tasks/<id>/SPIKE.md` written, deciding all five questions above: the v1
  feature cut, the named MCP control-tool set mapped to endpoints, the
  orchestrator-only scoping mechanism, the keep/drop/relocate call for the
  existing 8 tools, and the Telegram transport/auth/rendering approach.
  (manual: user confirms the direction)
- The chosen decomposition seeded as tatr tasks, each tagged and priority-slotted
  relative to the current backlog. (cmd: `tatr ls -f ':tags contains telegram'`)
- No shipped app code - this is research and direction only. (manual: diff is
  SPIKE.md + task files)

## Notes

- Grounding pointers:
  - 8 current tools + `apply_disabled_tools`/`SCUFRIS_DISABLED_TOOLS`:
    `scufris/mcp_server.py`.
  - MCP wiring (registers scufris server for every agent, global settings):
    `scufris/agent.py:_mcp_overrides` / `_server_override`.
  - Reserved synthetic orchestrator (`ORCHESTRATOR_ID`, hidden default, configured
    from the settings store not agents.json): `scufris/agent_store.py`.
  - Control surface to wrap as tools: the routes in `scufris/app.py`.
  - Backend abstraction (codex/claude) the orchestrator drives: `scufris/backends.py`.
- Old project for reference: `github.com/alexjercan/scufris-bot` (Telegram bot
  patterns - transport, command set, session handling).
- Relates to / likely depends on: U1 "orchestrator as a first-class hidden,
  editable agent" (`tasks/20260721-234558`) and the editable-tools settings task
  (`tasks/20260720-184137`) - the orchestrator-only scoping should be consistent
  with how the orchestrator is already special-cased.
- Follow the harness-first testing rule: T2/T4/T5 ship integration tests
  (stub Telegram with respx, stub the backend) and an `examples/` script, not
  isolated unit tests.
