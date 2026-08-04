# Read-only per-project skills+tools discovery + endpoint (provider-aware)

- PRIORITY: 25
- TAGS: feature, agents, backend, projects
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As an operator, I want scufris to DISCOVER the skills and custom tools a
project defines in its own working tree (its `.claude/skills`, its `.mcp.json`
/ settings MCP servers - and the codex equivalents), and serve them read-only
for a given agent, so the agent settings page (task 20260723-225621) can show
what recipes and tools that agent can be steered toward. Read-only: no create /
edit / delete of anything on disk.

This is the backend half: a provider-aware discovery module (mirroring the
`read_project_tasks` pattern in `scufris/projects.py`) plus one HTTP endpoint.

## Steps

- [x] Add `scufris/project_capabilities.py` with Pydantic models:
      `ProjectSkill{name: str, description: str, source: str}` (source = the
      SKILL.md path relative to cwd), `ProjectTool{name: str, description: str,
      source: str, kind: str}` (kind = transport: "stdio"/"http"/"sse"/"ws", or
      "" if unknown; description = a short summary e.g. the command or url),
      and `ProjectCapabilities{skills: list[ProjectSkill], tools:
      list[ProjectTool]}`.
- [x] Define a provider registry `_PROVIDER_SOURCES: dict[str, ProviderSources]`
      keyed by CANONICAL backend (`canonical_backend` from config.py), so
      discovery paths are data-driven and easy to extend (DoD item 5). Ground
      truth (confirmed, see DECISION.md):
      - `claude`: skill dirs `[".claude/skills"]`; MCP JSON files `[".mcp.json",
        ".claude/settings.json", ".claude/settings.local.json"]` (top-level
        `mcpServers` object).
      - `codex`: skill dirs `[".codex/skills"]`; MCP TOML files
        `[".codex/config.toml"]` (`[mcp_servers.<name>]` tables).
      - other/unknown backend -> empty ProviderSources (no discovery).
- [x] `read_project_skills(cwd, backend) -> list[ProjectSkill]`: for each skill
      dir, if it exists, glob `*/SKILL.md`; parse each file's YAML frontmatter
      (the `---`-delimited head) for `name` and `description`. Write a MINIMAL
      frontmatter parser (no PyYAML dep - it is not installed): read lines
      between the first two `---` fences, split each on the first `:`. `name`
      defaults to the skill directory name when absent; `description` defaults
      to "". Tolerant: a missing dir, unreadable file, or absent frontmatter
      yields no entry / empty fields, never raises. Sort by name.
- [x] `read_project_tools(cwd, backend) -> list[ProjectTool]`: parse each JSON
      MCP file's top-level `mcpServers` object (name -> spec) and each TOML
      file's `[mcp_servers.*]` tables (use stdlib `tomllib`, available on 3.13).
      For each server derive `kind` from an explicit `type` field, else "stdio"
      when a `command` is present or "http" when a `url` is present, else "";
      `description` = the command (+ first arg) or the url, best-effort; `source`
      = the file path relative to cwd. Merge across files, de-duplicating by
      server name (first file wins, in registry order). Tolerant: missing /
      malformed files are skipped with a log line, never raise. Sort by name.
- [x] `read_project_capabilities(cwd, backend) -> ProjectCapabilities`:
      combine the two. This is the single entry point the endpoint calls.
- [x] In `scufris/app.py`, add `GET /api/agents/{agent_id}/capabilities` ->
      `ProjectCapabilities`, in the agents route family (near
      `get_agent_scoped_tools` ~L1551). Resolve the agent via `_require_agent`
      (404 if missing) and its project via `_require_agent_project`; when there
      is no bound project (orchestrator / project-less agent) return an EMPTY
      `ProjectCapabilities()`. Otherwise call `read_project_capabilities(
      project.cwd, canonical_backend(agent.backend))`. Read-only (GET only).
- [x] Re-export the models from `scufris/app.py`'s imports as needed and add the
      endpoint under the existing `agents` tag so it shows in the OpenAPI docs.
- [x] Write `tasks/20260723-225616/DECISION.md` recording the per-provider
      discovery paths + the "read-only, list-only" scope choice (see the plan
      skill's DECISION format), and add a pointer line to the umbrella
      `GOAL.md` Decisions index.
- [x] Add tests (mirror the existing projects/app tests): a
      `tests/test_project_capabilities.py` that builds a temp project tree with
      `.claude/skills/foo/SKILL.md` (frontmatter), a `.mcp.json`, and a
      `.claude/settings.json`, and asserts the discovered skills+tools; plus a
      codex-tree case with `.codex/skills` + `.codex/config.toml`; plus
      tolerance cases (missing dirs, malformed frontmatter/JSON/TOML -> empty,
      no raise). Add a FastAPI endpoint test (mirror existing app tests) for:
      populated project agent, orchestrator/project-less -> empty, unknown agent
      -> 404.

## Definition of Done

- `read_project_capabilities` discovers a project's `.claude/skills/*/SKILL.md`
  skills (name+description) tolerant of missing/malformed input, never raising
  (test: `test_project_capabilities.py` skills cases incl. malformed).
- It discovers MCP servers from `.mcp.json` + `.claude/settings.json` (claude)
  and `.codex/config.toml` (codex), merged and de-duplicated, with a `kind`
  transport (test: `test_project_capabilities.py` tools cases for both
  providers).
- `GET /api/agents/{id}/capabilities` returns the discovered capabilities for a
  project agent, an empty set for the orchestrator/project-less agent, and 404
  for an unknown agent (test: the endpoint test cases).
- Discovery paths are provider-aware via a documented registry keyed by
  canonical backend (cmd: `grep -n "_PROVIDER_SOURCES" scufris/project_capabilities.py`).
- A `DECISION.md` records the per-provider paths + read-only scope, indexed in
  GOAL.md (cmd: `grep -n "DECISION" tasks/20260723-225437/GOAL.md`).
- The Python check suite passes (cmd: run the repo's check gate - ruff format,
  ruff, mypy, pytest - per AGENTS.md).

## Notes

- Relevant files: `scufris/projects.py` (`read_project_tasks` at the bottom is
  the exact pattern to mirror - cwd-scoped, guarded by dir existence, never
  raises, logs on failure), `scufris/app.py` (`_require_agent` /
  `_require_agent_project` at ~L1107, `get_agent_scoped_tools` at ~L1551,
  `AgentTool`/`ToolParam` models at ~L247, the route-family ordering comment at
  ~L165 - `/capabilities` is a static suffix so it does not collide with
  `/{agent_id}`), `scufris/config.py` (`canonical_backend` at L237,
  `McpServerSpec` at L24 for the MCP entry shape).
- Ground truth on paths/schemas comes from a research pass (claude-code-guide,
  20260723): Claude project skills `.claude/skills/<name>/SKILL.md` + MCP from
  `.mcp.json` and `.claude/settings.json` (JSON `mcpServers`, entry `{type?,
  command, args, env}` or `{type:"http|sse|ws", url, headers}`, no type =
  stdio); Codex project skills `.codex/skills/<name>/SKILL.md` + MCP from
  `.codex/config.toml` (`[mcp_servers.<name>]`, command/args/env or url/type).
  Both SKILL.md use the same YAML frontmatter (name, description). Record this
  in DECISION.md.
- Deliberately NOT discovered this cycle: global (`~/.claude`, `~/.codex`)
  skills/servers - only PROJECT-tree sources, matching the goal's per-project
  scope; the emerging `.agents/skills` path (low confidence, undocumented in
  Claude Code) - note it in DECISION.md as a future extension, do not scan it.
- No PyYAML in deps (confirmed); `tomllib` IS stdlib on 3.13 (confirmed). Do not
  add dependencies - hand-parse frontmatter, use tomllib for TOML.
- Trust/security: cwd is a registered project dir (already trusted by the
  operator who registered it); discovery only READS files under that tree, no
  path escapes (glob is confined to the skill dirs under cwd). No execution of
  any discovered command.

## Outcome (CLOSED)

Added `scufris/project_capabilities.py` (provider-aware, read-only discovery of
a project's SKILL.md skills + MCP-server "tools") and
`GET /api/agents/{id}/capabilities`, wired to the agent's bound project cwd and
`canonical_backend(agent.backend)`. Orchestrator / project-less agent -> empty;
unknown agent -> 404. DECISION.md records the per-provider paths and the
read-only/list-only scope; indexed in GOAL.md.

What changed and why:
- A data-driven `_PROVIDER_SOURCES` registry (claude/codex) keyed by canonical
  backend, so a new provider is one dict entry (GOAL done-item 5). Mirrors the
  `read_project_tasks` tolerance pattern: cwd-scoped, guarded by existence,
  never raises, logs on malformed input.
- Minimal `---`-fence frontmatter parser instead of adding PyYAML; stdlib
  `tomllib` for codex config. See DECISION.md alternatives.
- MCP servers rendered per-SERVER (name + transport kind + command/url), not by
  launching them - the read-only/no-execution guarantee. `env` never surfaced.

Tests: `tests/test_project_capabilities.py` (11 cases: claude+codex skills,
merged/deduped tools for both providers, malformed/missing tolerance,
unknown-backend empty, combined entry point) + `test_agent_capabilities_endpoint`
in test_app.py (populated project agent, empty orchestrator, 404). All green.
ruff format + ruff check + mypy clean.

Difficulty / inherited red: the full suite has ONE failure,
`test_agent_config_omits_builtin_server_when_tools_disabled`, which is
PRE-EXISTING on master and INDEPENDENT of this change - it constructs `Settings`
without isolating `state_dir`, so it reads the real
`~/.local/state/scufris/settings.json` (which had `agent_tools_enabled: true`)
and the override wins over the constructor arg. Diagnosed by running the test in
isolation on master (fails identically) and inspecting the override store. Filed
as task 20260723-233337. The rest of the suite is green
(`pytest --deselect <that test>` exits 0). Per the repo's "green means adds no
NEW errors on a red baseline" doctrine, this change adds none.

Self-reflection: researching the codex/claude paths up front (before writing the
discovery layer) paid off - the registry matched the real conventions with no
rework. Next time, run the full suite on the pristine base BEFORE starting so an
inherited red is known from minute one rather than diagnosed at verify time.
