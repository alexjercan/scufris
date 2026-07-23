# Render read-only project skills+tools cards on the agent settings page

- STATUS: OPEN
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

- [ ] Add `ProjectSkill` and `ProjectTool` (or a single `ProjectCapabilities`)
      TypeScript interface to `web/src/common.ts`, mirroring the backend
      Pydantic models exactly (field names + optionality). Skill: `{name,
      description, source}`. Tool: `{name, description, source, kind}` (kind =
      the transport, e.g. "stdio"/"http", best-effort).
- [ ] In `web/src/agent-settings-view.ts`, add the fetch of the new endpoint
      (e.g. `/api/agents/<id>/capabilities`) into the `agentSettingsDeps.load`
      Promise.all fan-out, using the existing `maybe<T>()` best-effort helper so
      a failure never blanks the page. Skip the fetch for the orchestrator
      (`isOrchestrator`) and agents with no project - resolve to null there, the
      same pattern as `agentTools`.
- [ ] Thread the fetched capabilities through `AgentSettingsData` (a new
      `capabilities: ProjectCapabilities | null` field), defaulting to null.
- [ ] Add two read-only render functions mirroring `agentToolsPanel`: a
      "Project skills (N)" card and a "Project tools (N)" card. Each row shows
      name + description; a tool row also shows its source/kind. Follow the
      existing `settings__card` / `settings__title` / `settings__row` markup and
      `escapeHtml` every interpolated value.
- [ ] Empty states: when the project defines no skills (or no tools), render a
      clear "none" note card (mirroring `agentToolsPanel`'s empty branch) rather
      than omitting the card, so the surface is always transparent. When there
      is no bound project at all (orchestrator / project-less agent), render
      NEITHER card (capabilities is null) - do not show empty project cards for
      an agent that has no project.
- [ ] Place the two cards in `renderAgentSettings` next to the existing
      `agentToolsPanel` (inside the `agent.id !== ORCHESTRATOR_ID` block), after
      the agent's own tool surface, so project-scoped info groups together.
- [ ] Add a vitest render test in `web/src/agent-settings-view.test.ts`
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
