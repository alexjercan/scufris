# EPIC: Build a capability-based agent and plugin platform

- STATUS: OPEN
- PRIORITY: 0
- TAGS: goal,epic,backlog,agents,plugins

## Epic

Evolve Scufris into a capability-based personal agent workbench. Agents become
reviewable blueprints composed from a harness, model, instructions, workspace,
skills, MCP/plugin capabilities, permissions, memory policy, and expected
artifacts. Plugins package capability, policy, configuration, health, and UI
metadata around out-of-process integrations rather than arbitrary in-process
code.

## Done Means

1. Versioned agent blueprints can be created, validated, reused, and migrated
   across Codex, Claude, OpenCode, and mock
   (test: `test_agent_blueprint_roundtrip_and_migration`).
2. Plugins are discovered from manifests, validated, health-checked, and mapped
   to explicit capabilities without importing plugin code into Scufris
   (test: `test_plugin_manifest_discovery_and_health`).
3. Authentication, secret references, capability grants, approval policies,
   and an immutable action audit protect privileged and outward-facing tools
   (test: `test_plugin_action_requires_effective_grant_and_approval`).
4. The orchestrator can propose a specialized agent, but the user sees and can
   edit its harness, project, tools, permissions, and output before launch
   (test: `orchestrator-blueprint-approval.spec.ts`).
5. manual: creating a research, email-drafting, or presentation agent feels
   like configuring a clear specialist rather than editing raw backend flags.

## Child Tasks

- [ ] 20260729-102205 (p0, scufris) define the agent blueprint and plugin
      architecture
- [ ] 20260729-102206 (p0, scufris) add agent blueprint schemas and reusable
      templates
- [ ] 20260729-102207 (p0, scufris) add plugin manifests, discovery, and health
      reporting
- [ ] 20260729-102208 (p0, scufris) add local authentication and protected
      secret references
- [ ] 20260729-102919 (p0, scufris) add capability grants, approvals, and
      action audit
- [ ] 20260729-102209 (p0, scufris) let the orchestrator propose, approve, and
      launch agent blueprints

## Decisions

- Pending 20260729-102205 SPIKE.md and DECISION.md: blueprint composition,
  plugin packaging, process boundary, manifest versioning, and MCP relationship.
- Pending 20260729-102208 DECISION.md: local authentication and secret-storage
  boundary.
- Pending 20260729-102919 DECISION.md: capability and approval policy model.

## Manual Acceptance

- (pending) 20260729-102209: the proposal/approval UI makes every enabled tool
  and privileged capability obvious before launch.

## Flow State

- FLOW STEP: PLANNING
