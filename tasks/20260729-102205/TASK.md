# Spike: define the agent blueprint and plugin architecture

- STATUS: OPEN
- PRIORITY: 0
- TAGS: spike,backlog,agents,plugins,mcp

## Story

As a platform designer, I want a durable architecture for configurable agents
and extensions, so that Scufris can grow beyond project coding without binding
the product to one harness, one prompt format, or unsafe plugin execution.

## Steps

- [ ] Inventory the existing `AgentRecord`, backend abstraction, MCP catalogs,
      skills discovery, permissions, settings, and orchestrator creation flow.
- [ ] Define the blueprint fields and composition rules for harness/model,
      prompt/skills, workspace, plugins/MCPs, permissions, memory, limits, and
      expected output artifacts.
- [ ] Define plugin manifest fields for identity/version, tools/resources/
      prompts, config schema, secret references, capabilities, health probe,
      process transport, and optional UI/viewer contributions.
- [ ] Compare manifest-plus-MCP out-of-process plugins with in-process Python
      entry points and document the security, upgrade, failure, and packaging
      tradeoffs.
- [ ] Define compatibility/versioning, installation/discovery, trust levels,
      unavailable-plugin behavior, and backend capability negotiation.
- [ ] Define the minimum conformance contract a future harness must pass before
      becoming selectable.
- [ ] Write `SPIKE.md`, record accepted choices in `DECISION.md`, and refine
      the remaining epic children against that decision.

## Definition of Done

- The spike contains concrete blueprint and plugin manifest examples
  (cmd: `rg -n "harness|workspace|capabilities|secrets|health|artifacts" tasks/20260729-102205/SPIKE.md`).
- Plugin process isolation and the MCP-versus-plugin boundary are decided
  (cmd: `test -f tasks/20260729-102205/SPIKE.md && test -f tasks/20260729-102205/DECISION.md && tatr check 20260729-102205`).
- Existing agents and all current backends have an explicit migration/
  compatibility path
  (cmd: `rg -n "Codex|Claude|OpenCode|mock|migration" tasks/20260729-102205/SPIKE.md`).
- manual: the user accepts the blueprint and plugin model before platform
  implementation starts.

## Notes

- Epic: 20260729-102204.
- Prefer MCP as the capability transport and a Scufris manifest as packaging,
  policy, configuration, health, and UI metadata.
- Do not select LangChain or another new harness without a demonstrated missing
  capability and a conformance plan.

## Flow State

- FLOW STEP: PLANNING
