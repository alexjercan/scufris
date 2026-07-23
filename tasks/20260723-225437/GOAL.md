# Goal: surface per-project skills + custom tools read-only in agent settings UI

- DATE: 20260723
- UMBRELLA TASK: 20260723-225437
- LANDING SCOPE: squash-merge each task to local `master` via `sprout land`; do
  NOT push (user's call). Standard per-repo flow.

## Goal

A project can define, in its own working tree, the skills and custom tools its
agents get: Claude Code reads `<cwd>/.claude/skills/*/SKILL.md` skills and MCP
servers from `<cwd>/.mcp.json` + `<cwd>/.claude/settings.json`; codex reads its
own equivalents (skills dir + config MCP). scufris agents are project-bound, so
the operator should be able to SEE those project-defined skills and tools on the
agent's settings page, without editing them. This run delivers a read-only
surface: the backend discovers a project's skills and custom tools/MCP servers
(provider-aware), an endpoint serves them for a given agent (via its bound
project cwd), and the agent settings page renders them as read-only cards.

Scope is READ-ONLY display only. Managing (add/remove/edit) skills and tools is
explicitly out of scope (that is the broader old task 20260720-195545); this run
narrows to "list first, safe".

## Done means

1. Backend discovers a project's `.claude/skills/*/SKILL.md` skills (name +
   description from YAML frontmatter), tolerant of missing dir / malformed
   frontmatter (never raises). (test: backend unit/integration test over a temp
   project tree)
2. Backend discovers a project's custom MCP servers/tools from `<cwd>/.mcp.json`
   and `<cwd>/.claude/settings.json` (mcpServers), merged, provider-aware, and
   the codex equivalents for the codex backend. (test: backend test over a temp
   project tree with both files)
3. An HTTP endpoint serves the discovered skills + tools for an agent, scoped to
   its bound project cwd (empty for the orchestrator / no project), read-only.
   (test: FastAPI endpoint test)
4. The agent settings page renders the project's skills and custom tools as
   read-only cards (name, description/source), with an empty state when the
   project defines none. (test: agent-settings-view vitest render test)
5. Discovery is provider-aware: the source paths are chosen by the agent's
   backend (claude vs codex), documented, and easy to extend. (cmd: see the
   DECISION.md recording the per-provider paths)

Overall: `npm run ci` (web) and the Python check suite both pass on `master`
after every task lands.

## Tasks

Updated as tasks land (one line per land).

- [x] 20260723-225616 (p25, scufris) Read-only per-project skills+tools
      discovery + endpoint (provider-aware)
      landed 79c0f34; 1 review round (out-of-context APPROVE, 2 no-change NITs);
      filed follow-up 20260723-233337 for a pre-existing test-isolation red.
- [x] 20260723-225621 (p23, web) Render read-only project skills+tools cards on
      the agent settings page (depends on 20260723-225616)
      landed 4d3d296; 1 review round (out-of-context APPROVE, 1 cosmetic NIT
      addressed in-branch).

## Decisions (load-bearing, architectural)

- 20260723-225616 DECISION.md: provider-aware per-project discovery paths
  (claude: .claude/skills + .mcp.json/.claude/settings*.json; codex:
  .codex/skills + .codex/config.toml) and read-only/list-only scope. (ACCEPTED)

## Manual acceptance (batched for the user at Finish)

- (pending) 20260723-225621: on a real project agent whose project has a
  `.claude/skills` dir and/or a `.mcp.json`, the agent settings page shows those
  skills and tools as read-only cards, and shows clear empty states otherwise.
