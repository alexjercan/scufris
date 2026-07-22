# Spike: Telegram frontend - orchestrator-as-the-whole-UI, MCP control tools, orchestrator-only tool scoping

- DATE: 20260722-221359
- STATUS: RECOMMENDED
- TAGS: spike, feature, agent, mcp, frontend, telegram

## Question

Should Scufris grow a Telegram frontend like the old
`github.com/alexjercan/scufris-bot`, and if so, what is the v1 feature cut and
the architecture? Concretely, five sub-questions:

1. What subset of the dashboard is worth a chat-only face?
2. Which control actions need MCP tools so the orchestrator can DO, not just observe?
3. How do we make those control tools available to the ORCHESTRATOR ONLY, when
   today every agent gets the same MCP tools from global settings?
4. Which of today's 8 MCP tools do we keep, drop, or relocate?
5. What Telegram transport / auth / rendering do we use?

A good answer names the v1 tool set, the scoping mechanism (grounded in the
actual run path), and the transport, concretely enough that `/plan` can expand
it without re-litigating.

## Context (grounded in the code)

- **The 8 tools** live in `scufris/mcp_server.py`: `host_stats`, `disk_usage`,
  `list_processes`, `tatr_ls`, `tatr_show`, `tatr_new`, `list_agents`,
  `agent_status`. The two agent tools are READ-ONLY (observe, not control).
- **No orchestrator scoping today.** `scufris/agent.py:_mcp_overrides(settings)`
  registers the scufris MCP server for the codex invocation off GLOBAL settings
  (`agent_tools_enabled`, `disabled_tools`). At the run call site
  (`app.py:_launch_agent_turn` -> `backend.stream`, ~line 1099) EVERY agent -
  orchestrator included - is passed the same closed-over global `settings`; the
  only per-agent differentiation is `cwd` (project dir), `session_id`, and
  `permission_mode`. So all agents get an identical tool set. The orchestrator is
  distinguished only by `agent.id == ORCHESTRATOR_ID` and by having no project
  (`_require_agent_project` returns None -> runs in the server cwd).
- **The claude backend does not wire the scufris MCP at all** -
  `_claude_stream_args` (`backends.py`) never adds it; only codex does. Any
  MCP-based plan is codex-first until claude gets a `--mcp-config`.
- **The MCP server is a separate process** spawned by codex; it does NOT share
  the dashboard's in-memory `Supervisor`. Today's read-only agent tools work
  around this by reading PERSISTED state (`AgentStore` + backend `read_status`).
  A control action that needs the LIVE supervisor (start/steer a run) therefore
  cannot poke in-memory state from the MCP subprocess - it must cross back into
  the app process.
- **The control surface already exists as HTTP routes** in `scufris/app.py`:
  projects (`GET/POST /api/projects`, `POST /api/projects/new`,
  `GET/PATCH/DELETE /api/projects/{id}`, `/tasks`), agents
  (`GET/POST /api/agents`, `GET/PATCH/DELETE /api/agents/{id}`, `.../run`,
  `.../chat`, `.../status`, `.../fork`, `.../transcript`, ...), host
  (`/api/stats`, `/api/processes`), and the orchestrator chat
  (`/api/chat`, `/api/chat/stream`, `/api/chat/reset`).
- **Single-session home already exists.** `agent_store` has
  `orchestrator_session_id` / `set_orchestrator_session` - the natural place to
  pin the one Telegram chat's session.
- **Stack:** FastAPI + uvicorn, httpx already a dep (no telegram lib), settings
  via pydantic-settings with `env_prefix="SCUFRIS_"`. App launches through
  `scufris/cli.py:main`.

## Options considered

### Q1 - Feature scope for v1
- **Full parity** (settings editing, profiles, tool enable/disable, memory/usage
  panels, project detail, tatr management) - large, and much of it is
  configuration better left to the web UI.
- **"See the box + run agents" cut (recommended):** host stats, list/inspect
  agents, create an agent on a project, run/steer it, list/create projects. This
  is the 80/20 of "drive the box from my phone".
- **Read-only cut** (observe only) - safe but does not deliver the "create agent
  for project" the user explicitly wants.

### Q2 - Control-tool set and how it reaches the app
- **In-process store access from the MCP subprocess** - works for file-backed
  writes (create project/agent) but CANNOT launch/steer a run (needs the live
  in-app `Supervisor`), so it only covers half the actions. Rejected as the
  primary mechanism.
- **Local HTTP API calls (recommended):** the control tools `httpx` the
  dashboard's own API at `127.0.0.1:<port>`. Crosses the process boundary
  cleanly, reuses every endpoint's validation/lifecycle, and covers launch/steer
  because the app process owns the supervisor. Needs the base URL in the MCP
  server env (the app already knows `settings.host/port`).
- v1 tools: `list_projects`, `create_project`, `create_agent`, `run_agent`,
  `message_agent` (steer), plus the existing `list_agents` / `agent_status`
  re-pointed at the same convention.

### Q3 - Orchestrator-only scoping mechanism
- **Runtime env gate in one server** - keep a single scufris server, gate the
  control tools behind `SCUFRIS_ORCHESTRATOR=1` (mirrors the existing
  `SCUFRIS_DISABLED_TOOLS` env). Least plumbing, but "orchestrator-only" lives in
  a runtime `if` inside a shared server.
- **Register scufris MCP only for the orchestrator (recommended).** Thread an
  `is_orchestrator` boolean from the run call site
  (`_launch_agent_turn` knows `agent.id == ORCHESTRATOR_ID`) through
  `backend.stream -> _stream_app_server -> _mcp_overrides`, and register the
  scufris server ONLY when `is_orchestrator`. This makes the model structural:
  regular agents get NOTHING from scufris and draw their tools from their own
  project `.config` / `.skills`, exactly the user's vision; the orchestrator gets
  the whole scufris server (host + observe + control). It is also a behavior
  change (today regular agents get the 8 tools), which the user explicitly wants.
- **Two separate MCP servers** (observe-server for all, control-server for
  orchestrator) - cleanest separation but more moving parts than needed once the
  decision is "regular agents get nothing from scufris".

### Q4 - Keep / drop / relocate the current 8
- `tatr_ls` / `tatr_show` / `tatr_new`: **drop as MCP tools.** `tatr` is a skill
  the orchestrator runs via `Bash`; a dedicated MCP wrapper is redundant once the
  scufris server is orchestrator-only. Caveat: `tatr_new` writes, and it existed
  precisely because the MCP subprocess runs OUTSIDE the model's read-only
  sandbox; the orchestrator must therefore run in a write-capable permission mode
  to create tasks via Bash. Acceptable - the orchestrator is the privileged
  session by design.
- `host_stats` / `disk_usage` / `list_processes`: **keep, now orchestrator-only**
  (they ride along because the whole server becomes orchestrator-scoped). These
  answer "how's the box" in Telegram.
- `list_agents` / `agent_status`: **keep**, joined by the new control tools.

### Q5 - Telegram transport / auth / rendering
- **`python-telegram-bot` / `aiogram` library** - batteries included, but a heavy
  new dependency for a small surface.
- **Thin async httpx long-poll client (recommended).** `getUpdates` long-poll
  against the Bot API using the httpx already in the tree - matches how this
  codebase hand-wires codex/claude, adds no heavy dep, and avoids exposing a
  public webhook on a single home box. Auth: an allowlist of chat ids in settings
  (`SCUFRIS_TELEGRAM_ALLOWED_CHAT_IDS`); anything else is ignored. Session: pin
  the one chat to `orchestrator_session_id`; `/new` resets, `/help` lists
  commands. Token: `SCUFRIS_TELEGRAM_BOT_TOKEN`. Rendering: one final message per
  turn with a "typing..." chat action while the orchestrator streams, and a short
  tool-summary line (full token streaming into edited messages is a later
  polish). Runs as a background asyncio task inside the app process (launched when
  a token is configured), calling the orchestrator through the SAME internal path
  as `/api/chat/stream` - so the bot needs no self-HTTP (only the MCP control
  tools do).
- **Webhook** transport - rejected for v1 (needs a public endpoint / TLS on a
  home box).

## Recommendation

Build the Telegram frontend as a second face on the orchestrator, in this shape:

1. **Make the scufris MCP server orchestrator-only** by threading `is_orchestrator`
   through the run path and registering it only for the orchestrator. Regular
   agents stop receiving it and rely on their own project tools.
2. **Add control tools** (`list_projects`, `create_project`, `create_agent`,
   `run_agent`, `message_agent`) that call the local dashboard HTTP API, curated
   and bounded like the existing `_run` tools. Codex-first; claude MCP wiring is a
   follow-up.
3. **Drop the `tatr_*` MCP tools** (orchestrator uses the tatr skill via Bash);
   keep the host tools, now orchestrator-scoped.
4. **Add a thin httpx long-poll Telegram client** that maps the single chat to
   `orchestrator_session_id`, gated by a chat-id allowlist, token via settings,
   launched in-process, replying one message per turn with a tool-summary line.
5. Ship an `examples/` script that boots the bot against a stubbed Bot API + mock
   backend, and an integration test (respx-stubbed Telegram + stubbed backend).

This delivers "talk to the box from Telegram, see host stats, and create/inspect
agents on projects", with the control tools withheld from ordinary agents.

## Open questions

- **Orchestrator backend.** Control tools reach codex only (claude backend has no
  MCP wiring). v1 assumes the orchestrator runs on codex; wiring claude
  `--mcp-config` is a follow-up (fold into the control-tools task notes, or a
  separate task if it grows).
- **Permission mode for the orchestrator.** Dropping `tatr_new` means task
  creation goes through Bash, which needs a write-capable sandbox. Confirm the
  orchestrator's default permission mode supports this.
- **Reply streaming fidelity.** One-message-per-turn is the v1 call; whether to
  invest in edited-message token streaming is a UX decision after the first
  hands-on use.
- **Deploy shape.** In-process asyncio task vs a separate `scufris telegram`
  process - v1 recommends in-process; revisit if it complicates the app lifecycle.

## Next steps

Direction-level tasks seeded for `/plan` to break into steps:

- tatr 20260722-222717: T1 - orchestrator-only tool scoping (thread is_orchestrator; register scufris MCP only for the orchestrator)
- tatr 20260722-222722: T2 - control MCP tools over the local HTTP API (list/create project, create/run/message agent)
- tatr 20260722-222729: T3 - prune the MCP surface (drop tatr_* tools; host tools stay, orchestrator-scoped; update steering/docs)
- tatr 20260722-222734: T4 - Telegram transport (httpx long-poll, chat->orchestrator session, auth allowlist, token config, in-process launch)
- tatr 20260722-222739: T5 - reply rendering + end-to-end example (final-per-turn + tool summary; examples/ bot script; respx integration test)

## Fix record

(Each implementing task appends a few lines here as it lands.)
