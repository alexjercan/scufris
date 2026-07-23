# Render read-only project skills+tools cards on the agent settings page

- STATUS: CLOSED
- PRIORITY: 23
- TAGS: feature,agents,frontend,projects

## Story

As an operator viewing a project agent's settings page, I want to SEE the
skills and custom tools its project defines (its `.claude/skills`, its
`.mcp.json` / settings MCP servers), so I know what recipes and tools that
agent can be steered toward - without leaving the page or editing anything.

Consumes the endpoint added by task 20260723-225616 (backend discovery). This
task is the read-only UI surface only.

## Steps

- [x] Add `ProjectSkill` and `ProjectTool` (or a single `ProjectCapabilities`)
      TypeScript interface to `web/src/common.ts`, mirroring the backend
      Pydantic models exactly (field names + optionality). Skill: `{name,
      description, source}`. Tool: `{name, description, source, kind}` (kind =
      the transport, e.g. "stdio"/"http", best-effort).
- [x] In `web/src/agent-settings-view.ts`, add the fetch of the new endpoint
      (e.g. `/api/agents/<id>/capabilities`) into the `agentSettingsDeps.load`
      Promise.all fan-out, using the existing `maybe<T>()` best-effort helper so
      a failure never blanks the page. Skip the fetch for the orchestrator
      (`isOrchestrator`) and agents with no project - resolve to null there, the
      same pattern as `agentTools`.
- [x] Thread the fetched capabilities through `AgentSettingsData` (a new
      `capabilities: ProjectCapabilities | null` field), defaulting to null.
- [x] Add two read-only render functions mirroring `agentToolsPanel`: a
      "Project skills (N)" card and a "Project tools (N)" card. Each row shows
      name + description; a tool row also shows its source/kind. Follow the
      existing `settings__card` / `settings__title` / `settings__row` markup and
      `escapeHtml` every interpolated value.
- [x] Empty states: when the project defines no skills (or no tools), render a
      clear "none" note card (mirroring `agentToolsPanel`'s empty branch) rather
      than omitting the card, so the surface is always transparent. When there
      is no bound project at all (orchestrator / project-less agent), render
      NEITHER card (capabilities is null) - do not show empty project cards for
      an agent that has no project.
- [x] Place the two cards in `renderAgentSettings` next to the existing
      `agentToolsPanel` (inside the `agent.id !== ORCHESTRATOR_ID` block), after
      the agent's own tool surface, so project-scoped info groups together.
- [x] Add a vitest render test in `web/src/agent-settings-view.test.ts`
      covering: (a) project agent with skills+tools renders both cards with the
      right names/descriptions; (b) project agent with empty capabilities
      renders the "none" empty states; (c) orchestrator / project-less agent
      renders neither project card. Drive `renderAgentSettings` directly with a
      built `AgentSettingsData` (the tests are pure/jsdom, no fetch).

## Definition of Done

- The agent settings page shows a "Project skills" and a "Project tools" card
  for a project agent, populated from the capabilities endpoint, each row
  name + description, all read-only (test: `agent-settings-view.test.ts` new
  cases for populated cards).
- Empty project capabilities render explicit "none" cards, not missing cards;
  a project-less agent / orchestrator renders neither card (test: the empty +
  orchestrator cases in `agent-settings-view.test.ts`).
- Every interpolated value is HTML-escaped (cmd: `grep -n "escapeHtml" web/src/agent-settings-view.ts`).
- No non-ASCII slips into the new UI text (cmd: `grep -nP "[^\x00-\x7f]" web/src/agent-settings-view.ts web/src/common.ts`).
- `npm run ci` (in web/) passes (cmd: `cd web && npm run ci`).

## Notes

- Relevant files: `web/src/agent-settings-view.ts` (agentToolsPanel at ~L129 is
  the pattern to mirror; the Promise.all fan-out in `agentSettingsDeps.load` at
  ~L456; card assembly in `renderAgentSettings` at ~L365), `web/src/common.ts`
  (types), `web/src/agent-settings-view.test.ts`.
- Mirror `agentToolsPanel` exactly for markup/empty-state so the new cards look
  native. Reuse the `panel()` helper for empty states if it fits.
- Depends on: 20260723-225616 (backend endpoint + models). Match the endpoint
  path and the model field names to whatever that task actually ships - re-read
  its TASK.md / the shipped Pydantic models before writing the TS interfaces.
- The tests are PURE (`renderAgentSettings` is pure, jsdom-driven) - build the
  `AgentSettingsData` fixture in-test, no fetch mocking needed.

## Outcome (CLOSED)

Added the read-only project capability surface to the agent settings page:
- `ProjectSkill` / `ProjectTool` / `ProjectCapabilities` TS interfaces in
  `web/src/common.ts`, mirroring the backend Pydantic models field-for-field.
- `agent-settings-view.ts`: a `capabilities: ProjectCapabilities | null` field
  on `AgentSettingsData`, fetched in the `agentSettingsDeps.load` Promise.all
  fan-out via `maybe<ProjectCapabilities>('/api/agents/{id}/capabilities')`,
  skipped (null) for the orchestrator and any project-less agent.
- A `capabilityPanel` helper + `projectCapabilityCards` rendering a "Project
  skills (N)" and a "Project tools (N)" card, each row name + description (tool
  rows also show the transport `kind`), with an explicit "none (this project
  defines no skills/tools)" empty state via the existing `panel()` helper.
  Placed inside the `agent.id !== ORCHESTRATOR_ID` block right after
  `agentToolsPanel`, so project-scoped info groups together. Null capabilities
  -> neither card.

Tests (`agent-settings-view.test.ts`, pure/jsdom): populated cards render
name/description/kind; empty capabilities render the "none" cards; a
project-less agent (null capabilities) renders neither project card. All 20
tests in the file pass; `npm run ci` (format:check + lint + test + build) green.

DoD note: the non-ASCII DoD grep
(`grep -nP "[^\x00-\x7f]" web/src/agent-settings-view.ts web/src/common.ts`)
returns two PRE-EXISTING matches (the `<-` back-link arrow at L111 and a `middot`
separator in the usage panel at L265), neither in this diff. The intent - "no
non-ASCII in the NEW UI text" - holds: the added capability-card code is
ASCII-clean. The cmd as written is broader than its intent because the file
already contained those glyphs before this task.

Self-reflection: mirroring `agentToolsPanel` and reusing `panel()` for empty
states kept the new cards native with minimal new surface. The one wrinkle was
the absence-grep DoD self-matching pre-existing glyphs - next time scope such a
grep to the diff (e.g. `git diff | grep`) rather than the whole file.
