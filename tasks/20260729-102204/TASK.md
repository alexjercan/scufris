# EPIC: Build a capability-based agent and plugin platform

- STATUS: OPEN
- PRIORITY: 0
- TAGS: goal,epic,backlog,agents,plugins
- KIND: EPIC
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT

## Epic

Evolve Scufris into a capability-based personal agent workbench. Agents are
created from reviewable presets composed from a harness, model, instructions,
workspace, skills, MCP/plugin capabilities, permissions, memory policy, and
expected artifacts. Plugins package capability, policy, configuration, health,
and UI metadata around out-of-process integrations rather than arbitrary
in-process code.

## Done Means

1. Reusable agent presets can be created, validated, resolved against project
   defaults, and mapped onto Codex, Claude, OpenCode, and mock
   (test: `test_agent_preset_roundtrip`).
2. Plugins are discovered from manifests, validated, health-checked, and mapped
   to explicit capabilities without importing plugin code into Scufris
   (test: `test_plugin_manifest_discovery_and_health`).
3. Authentication, secret references, capability grants, approval policies,
   and an immutable action audit protect privileged and outward-facing tools
   (test: `test_plugin_action_requires_effective_grant_and_approval`).
4. The orchestrator can propose a specialized agent, but the user sees and can
   edit its harness, project, tools, permissions, and output before launch
   (test: `orchestrator-agent-approval.spec.ts`).
5. manual: creating a research, email-drafting, or presentation agent feels
   like configuring a clear specialist rather than editing raw backend flags.

## Child Tasks

- [ ] 20260729-102205 (p62, v0.2.0) define the reusable agent preset
      architecture
- [ ] 20260729-102206 (p61, v0.2.0) add reusable agent preset schemas and
      templates
- [ ] 20260729-102207 (p0, backlog) add plugin manifests, discovery, and health
      reporting
- [ ] 20260729-102208 (p0, backlog) add protected secret references and
      redaction (the dashboard-authentication half was pulled forward into
      v0.1.0 as 20260729-125015)
- [ ] 20260729-102919 (p0, backlog) add capability grants, approvals, and
      action audit
- [ ] 20260729-102209 (p0, backlog) let the orchestrator propose, approve, and
      launch specialist agents

## Decisions

- Pending 20260729-102205 SPIKE.md and DECISION.md: minimum preset composition,
  project defaults/overrides, backend mapping, and capability-reference
  boundary. Plugin packaging remains owned by 20260729-102207.
- Pending 20260729-102208 DECISION.md: local authentication and secret-storage
  boundary.
- Pending 20260729-102919 DECISION.md: capability and approval policy model.

## Manual Acceptance

- (pending) 20260729-102209: the proposal/approval UI makes every enabled tool
  and privileged capability obvious before launch.

## Deferred (2026-07-29 backlog review)

This epic was scoped for a platform with many operators, many harnesses, and
untrusted extensions. Scufris has one operator, one host, and four backends, so
the following are explicitly OUT until something concrete needs them:

- Cross-backend blueprint MIGRATION. The harnesses differ in what they can do;
  portability is best-effort, not a done-criterion. (Removed from Done Means 1.)
- Plugin TRUST LEVELS and a conformance contract a future harness must pass.
  Speculative: there is no fifth backend and no third-party plugin author.
- Backend capability NEGOTIATION as a general mechanism.

What survives as the valuable slice, in this order: reusable agent presets
(blueprints minus versioning and migration), an MCP/plugin registry with health
(20260729-102207), and a single approval-plus-audit gate (20260729-102919).

The reusable-preset slice (20260729-102205 and 20260729-102206) is pulled into
v0.2.0 as a prerequisite for stage-specific planning/work/review agents. This
does not pull forward plugin manifests, secret references, general capability
grants, or orchestrator launch UI.

The approval/audit pattern is being proven FIRST, concretely, by the host
operator epic (20260729-124655, task 20260729-125029) against one real
consumer. Generalize from that working implementation rather than designing the
general system ahead of it.

## Sequencing

- Post-v0.1.0 order (2026-07-29 backlog review): THIRD of the five backlog
  epics, and only in its sliced form (see Deferred above). Re-plan the children
  against that slice before scheduling; the epic as originally written is a
  governance platform for a product with many operators and untrusted
  extensions, which this is not.
- The preset foundation is scheduled for v0.2.0; the remaining plugin/platform
  children stay backlog, so this mixed-schedule epic remains OPEN and tagged
  `backlog`.
