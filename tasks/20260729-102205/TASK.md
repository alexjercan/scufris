# Spike: define the reusable agent preset architecture

- STATUS: OPEN
- PRIORITY: 62
- TAGS: spike, v0.2.0, agents, backend
- KIND: SPIKE
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT

## Story

As an operator, I want the smallest durable model for reusable agent presets,
so that a project can choose planning, implementation, review, host, or other
specialists without re-entering backend flags and without binding the product
model to one harness's CLI.

## Steps

- [ ] Inventory the existing `AgentRecord`, backend abstraction, MCP catalogs,
      skills discovery, permissions, settings, and orchestrator creation flow.
- [ ] Define preset identity and composition for harness/backend, model,
      instructions, optional skills, workspace scope, permission mode,
      capability/tool references, memory/session policy, limits, and expected
      output contract.
- [ ] Define project defaults, stage-specific named presets, controlled
      overrides, and a fully resolved effective-preset view. Include concrete
      `plan`, `work`, and `review` examples that may select different backends.
- [ ] Define the boundary between a reusable preset, an instantiated agent, and
      an individual run/task assignment; align it with the relationships chosen
      by 20260729-220835.
- [ ] Map the effective preset onto Codex, Claude, OpenCode, and mock, preserving
      backend-specific differences rather than promising lossless portability.
- [ ] Decide how presets reference today's built-in MCP/skill capabilities and
      how unavailable references fail before launch. Leave plugin discovery,
      process packaging, trust levels, and general capability negotiation to
      20260729-102207/20260729-102919.
- [ ] Define the migration/compatibility path from current `AgentRecord`
      configuration without requiring cross-backend preset migration.
- [ ] Write `SPIKE.md`, record accepted choices in `DECISION.md`, and refine
      20260729-102206 against that decision.

## Definition of Done

- The spike contains concrete planning, implementation, and review presets with
  harness, workspace, permissions, skills/capabilities, and output contracts
  (cmd: `rg -n "plan|work|review|harness|workspace|permission|capabilit|output" tasks/20260729-102205/SPIKE.md`).
- Preset, instantiated-agent, run-assignment, project-default, and override
  boundaries are decided
  (cmd: `test -f tasks/20260729-102205/SPIKE.md && test -f tasks/20260729-102205/DECISION.md && tatr check 20260729-102205`).
- Current AgentRecords and all current backends have an explicit, honest
  compatibility path
  (cmd: `rg -n "AgentRecord|Codex|Claude|OpenCode|mock|compatib" tasks/20260729-102205/SPIKE.md`).
- The user accepts the preset model before schema implementation starts (manual: user check).

## Notes

- Epic: 20260729-102204.
- Depends on: 20260729-220835 for the template/agent/run and flow-stage
  relationships.
- V0.2.0 scope is reusable presets only. Plugin manifests, process isolation,
  secret references, trust, health catalog, and general approval policy remain
  in their existing backlog tasks.
- Prefer references to the existing MCP/skill catalogs over embedding one
  backend's raw CLI arguments in a preset.
- Do not select LangChain or another new harness without a demonstrated missing
  capability and a conformance plan.
- Do not spend spike effort on cross-backend migration, plugin trust levels,
  backend capability negotiation, or a hypothetical future-harness conformance
  suite. See the Deferred section of 20260729-102204.
