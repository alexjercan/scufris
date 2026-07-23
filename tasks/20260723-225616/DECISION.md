# Decision: provider-aware per-project skills+tools discovery, read-only

- DATE: 20260723-225616
- STATUS: ACCEPTED
- TASK: 20260723-225616
- TAGS: decision, agents, projects, mcp

## Context

A scufris agent is bound to a project (a cwd). Both providers we run (Claude
Code and codex) let a project define, IN ITS OWN WORKING TREE, the skills and
MCP tools its agents get - but they use DIFFERENT on-disk conventions and
DIFFERENT serialization (JSON vs TOML). We want to SURFACE those per-project
skills and tools on the agent settings page so the operator can see what an
agent can be steered toward. The scope for this run is display only.

Paths/schemas were established by a research pass (claude-code-guide,
20260723), not from memory - see the umbrella GOAL.md and the backend TASK.md
Notes for the source list.

## Decision

Discover per-project capabilities PROVIDER-AWARE, driven by a data registry
`_PROVIDER_SOURCES` keyed by `canonical_backend(agent.backend)` in
`scufris/project_capabilities.py`:

- **claude**: skills from `<cwd>/.claude/skills/<name>/SKILL.md`; MCP servers
  from `<cwd>/.mcp.json`, `<cwd>/.claude/settings.json`,
  `<cwd>/.claude/settings.local.json` (top-level JSON `mcpServers` object).
- **codex**: skills from `<cwd>/.codex/skills/<name>/SKILL.md`; MCP servers from
  `<cwd>/.codex/config.toml` (`[mcp_servers.<name>]` TOML tables).
- any other backend (opencode/mock/unknown): empty sources -> no discovery.

SKILL.md frontmatter is parsed with a minimal hand-rolled `---`-fence parser
(no PyYAML dependency); codex TOML with stdlib `tomllib`. MCP servers merge
across a backend's files in registry order, de-duplicating by name (first file
wins). An MCP server's `kind` is its declared `type`, else inferred "stdio"
(has `command`) or "http" (has `url`).

The whole surface is READ-ONLY and LIST-ONLY: discovery only READS files under
the (already operator-trusted) project cwd, never writes, and never executes a
discovered command. `env` values are deliberately never surfaced (may hold
secrets). Managing (add/remove/edit) skills or tools is explicitly out of scope
(that is the broader task 20260720-195545).

## Alternatives considered

- **One hard-coded Claude path set.** Simpler, but wrong for codex agents (whose
  projects use `.codex/*`), and scufris runs both backends first-class. The
  registry costs one dict entry per provider and keeps discovery honest per
  agent.
- **Scan global dirs too (`~/.claude`, `~/.codex`).** Rejected: the goal is
  PER-PROJECT capabilities; global skills are the operator's environment, not a
  property of the project the agent runs in. Out of scope this run.
- **Scan the emerging `.agents/skills` path.** Rejected for now: low confidence
  / undocumented in Claude Code (per the research pass). Left as a future
  extension - add a `skill_dirs` entry when it firms up.
- **Launch each MCP server to enumerate its individual tools.** Rejected:
  executing project-declared commands violates the read-only/no-execution
  guarantee. One card per server (name + transport + command/url) is the safe,
  honest surface.
- **Add PyYAML for frontmatter.** Rejected: a new dependency for two
  single-line keys. A 10-line fence parser covers the SKILL.md format.

## Consequences

- Adding a provider = one `_PROVIDER_SOURCES` entry, no new code paths.
- The surface is safe by construction: read-only, no command execution, secrets
  (`env`) never rendered.
- The minimal frontmatter parser only reads flat single-line `key: value`
  pairs; a multi-line YAML value would be truncated. Acceptable - `name` and
  `description` are single-line by the SKILL.md convention; if a richer field is
  ever needed, swap in a real YAML parser then.
- Tool cards show the SERVER, not its individual tools; enumerating the tools
  would require running the server, which this design forbids.
