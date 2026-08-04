# Telegram read-only /settings subcommands + /stats

- PRIORITY: 55
- TAGS: feature, telegram, agents, frontend, backend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

# Read-only /settings subcommands + /stats for the Telegram bot

## Story

As the single allowed Telegram user, I want read-only `/settings` subcommands
and a `/stats` command that mirror the web dashboard's orchestrator data, so I
can check the agent's health, usage, and tools plus the host's health from
chat - without opening the browser. This is the read-only first slice of
"configure the orchestrator from Telegram"; writes and remote command
execution are explicitly out of scope for now.

Command surface (confirmed with user):

- `/settings` (no arg) -> a compact summary: backend, model, auth mode,
  enabled, permission mode, tool count (enabled/total), primary usage %, and
  the worst health status.
- `/settings health` -> the Health card: scufris version, backend + version,
  session count, and each diagnostic check (ok/warn/error with hint).
- `/settings usage` -> the account usage/quota: plan, primary/secondary rate
  window used % and reset.
- `/settings tools` -> the orchestrator tool catalog: total count + tools
  grouped by server (scufris/den), disabled tools marked.
- `/settings <unknown>` -> a short usage line listing the valid subcommands.
- `/stats` -> a COMPACT host health snapshot (one tidy message: host+uptime,
  CPU% + load, mem%/swap, disk % per mount, net up/down rate, top temp, GPU
  line when present).

These are quick reads, not orchestrator turns: they bypass the turn/busy
machinery and answer immediately.

## Design

Follows the established Telegram conventions:

- The bot stays a thin transport that drives everything through injected
  callbacks (no self-HTTP), and RENDERING lives in `telegram.py` (like
  `render_reply`/`markdown_reply`/`_format_tool`).
- Data providers are built app-side (in `create_app`, where `collector`,
  `agents`, `settings`, `agent_health`, `read_usage`, `_tools_for_servers`
  are in scope) and return domain models; `telegram.py` renders them.

Breaking the app<->telegram type cycle: `ToolParam`, `AgentTool`, and
`McpServerHealth` currently live in `app.py` (which imports `telegram`), so
`telegram.py` cannot import `AgentTool` for the tools render without a cycle.
Move those three DTOs to a neutral module (`mcp_common.py`, which imports no
app code) and re-export them from `app.py` so `app.X` and the OpenAPI/JSON
shape are unchanged. `AgentHealth` (health.py), `UsageQuota` (sessions.py) and
`HostStats` (metrics.py) are already neutral.

Provider bundle: a frozen `SettingsOps` dataclass in `telegram.py` holding the
async provider callables, injected into `TelegramBot.__init__` alongside the
existing callbacks:

- `health() -> AgentHealth` via `agent_health(settings, is_orchestrator=True)`
  (after `_ensure_den_path`).
- `usage() -> UsageQuota | None` via `read_usage(resolve_codex_home(settings))`,
  gated on `settings.agent_enabled` (None when disabled).
- `tools() -> list[AgentTool]` via
  `_tools_for_servers(_mcp_servers_for_audience(ORCHESTRATOR_ID))`.
- `stats() -> HostStats` via `collector.sample()` (wrap the sync call).
- `info() -> OrchestratorInfo` - a small NEUTRAL dataclass (defined in
  telegram.py) carrying backend, model, auth_mode, enabled, permission_mode
  for the summary's config line (avoids importing `AgentConfig`).

Rendering: reuse the file's emoji-via-`\N{...}` + guarded-send conventions.
The bodies are structured read-outs; render them as MarkdownV2 (monospace code
block for the aligned stats/health tables) via the existing
`telegramify_markdown` path, or plain text with the same
`_send_reply`-style plain-text fallback so a formatting error never drops the
reply.

## Steps

- [x] Move `ToolParam`, `AgentTool`, `McpServerHealth` from `app.py` to a
      neutral module and re-export from `app.py` (keeps `app.McpServerHealth`
      etc. resolving; OpenAPI unchanged). Landed as a new `scufris/mcp_models.py`
      rather than `mcp_common.py` - a dedicated DTO home reads clearer than
      overloading the MCP-plumbing helpers module.
- [x] Add pure render functions in `telegram.py`: `render_settings_summary`,
      `render_health`, `render_usage`, `render_tools`, `render_stats`. The
      unknown-subcommand case is a fixed `SETTINGS_USAGE` constant (a plain
      message), not a renderer. Source stays ASCII (emoji as `\N{...}`); each
      renderer handles empty/None/degraded inputs.
- [x] Add the neutral `OrchestratorInfo` dataclass and the frozen `SettingsOps`
      bundle in `telegram.py`; add keyword-only `settings_ops: SettingsOps` to
      `TelegramBot.__init__`.
- [x] Extend `_dispatch` (with a `_command_arg` split) to route `/settings [sub]`
      and `/stats`, calling the providers and sending the rendered body via a
      guarded send (MarkdownV2 -> plain fallback). These bypass the turn/busy path.
- [x] Update `HELP_TEXT` to document `/settings [health|usage|tools]` and
      `/stats`.
- [x] Build the providers app-side (`_build_telegram_settings_ops` in
      `create_app`) wiring the in-process readers, and pass `SettingsOps` into
      `TelegramBot`.
- [x] Tests in `tests/test_telegram.py`: pure render-function tests for each
      renderer (populated, empty/None, degraded/warn inputs); dispatch tests
      with a fake `SettingsOps` + respx-stubbed `sendMessage` for `/settings`,
      `/settings health|usage|tools`, `/settings bogus`, and `/stats`; assert
      `/help` includes the new commands. Follows the existing fake-callback +
      respx pattern; the existing bot constructors updated for the new param.
- [x] Docs sync: updated the `telegram.py` module docstring for the new command
      surface. The README has no Telegram command list, so nothing to change there.

## Definition of Done

1. `/settings`, `/settings health`, `/settings usage`, `/settings tools`, and
   `/stats` render orchestrator/host data in Telegram, driven by in-process
   providers with no self-HTTP.
   (test: `python -m pytest tests/test_telegram.py`)
2. Render functions are pure and unit-tested for populated, empty/None, and
   degraded inputs.
   (test: `python -m pytest tests/test_telegram.py -k render`)
3. `/help` lists the new commands.
   (test: dispatch test asserts `HELP_TEXT` contains `/settings` and `/stats`)
4. Unknown `/settings <x>` returns a usage line, not an error or a turn.
   (test: `python -m pytest tests/test_telegram.py -k settings`)
5. Read-only only: no config writes, tool toggles, or remote command execution
   added.
   (manual: skim the diff for any PATCH/write/subprocess-exec path)
6. Full QA gate green (ruff + mypy + pytest), DTO relocation leaves the API
   shape unchanged.
   (cmd: `nix flake check`)

## Notes

- Decisions: (a) keep the injected-callback + render-in-telegram.py
  convention; (b) relocate the three tool DTOs to a neutral module to break the
  app<->telegram import cycle; (c) read-only scope; (d) compact `/stats`;
  (e) `/settings <sub>` subcommand surface - all per the user's plan-gate
  answers.
- Deferred (future tasks, not this one): config writes / tool enable-disable
  from Telegram, and remote command execution (home-manager / nixos-rebuild /
  running things on the host).

## Work Record

What changed:

- New `scufris/mcp_models.py` holds the three tool DTOs (`ToolParam`,
  `AgentTool`, `McpServerHealth`); `app.py` imports and re-exports them, so the
  API/OpenAPI shape is byte-identical. This breaks the would-be
  `telegram -> app` import cycle: the bot renders `AgentTool` without importing
  `app` (which imports `telegram`).
- `telegram.py` gained the read-only surface: an `OrchestratorInfo` neutral
  dataclass, a frozen `SettingsOps` provider bundle, five pure `render_*`
  functions (each a bold title over a fenced monospace block), a
  `settings_markdown` MarkdownV2 converter with a plain-text fallback, a
  `_command_arg` split, and `_dispatch` routes for `/settings [sub]` + `/stats`
  that bypass the turn/busy machinery. `HELP_TEXT` and the module docstring
  updated.
- `app.py` builds the providers in `_build_telegram_settings_ops` (orchestrator
  -scoped) from the SAME in-process readers the web endpoints use -
  `agent_health(is_orchestrator=True)`, `read_usage(resolve_codex_home(...))`
  (gated on `agent_enabled`), `_tools_for_servers(_mcp_servers_for_audience(...))`,
  and `collector.sample()` - and passes them into `TelegramBot`.

Alternatives considered:

- Bot self-HTTP to `/api/...` instead of injected providers: rejected to keep
  the established "transport only, no self-HTTP" design and its unit-testability.
- A single `/settings` dump vs. subcommands: the user chose `/settings <sub>`
  at the plan gate.
- Rendering in `app.py` (returning strings) vs. in `telegram.py` (returning
  models): kept rendering in `telegram.py`, matching `render_reply` and letting
  the render functions be pure-unit-tested.

Difficulties:

- Two test-data artifacts surfaced as false failures while writing the render
  tests (a `backend_version` that redundantly repeated "codex", and asserting no
  backtick in a body whose code fence legitimately uses backticks); both were
  test-side fixes, not render bugs.

Self-reflection:

- Making `settings_ops` a required keyword-only param was the right call for a
  clean contract, at the cost of touching all six constructor sites; a shared
  `_fake_settings_ops()` kept the test churn contained. Next time, front-load a
  grep of all constructor sites before changing a widely-built ctor so the
  update is one pass, not a re-run after a missed `idle_cancel` site.
