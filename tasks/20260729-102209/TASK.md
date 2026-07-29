# Let the orchestrator propose approve and launch agent blueprints

- STATUS: OPEN
- PRIORITY: 0
- TAGS: feature,backlog,agents,orchestrator,frontend

## Story

As a Scufris user, I want to describe an outcome and receive a transparent,
editable specialist-agent proposal, so that the orchestrator can assemble the
right harness, tools, permissions, and output contract without silently
enabling capabilities.

## Steps

- [ ] Add a structured orchestrator tool that proposes a blueprint from the
      available templates, backends, plugins, project context, and policies.
- [ ] Validate proposals server-side and explain missing, incompatible,
      unhealthy, or ungranted capabilities.
- [ ] Build a review surface showing name/purpose, harness/model, project root,
      instructions/skills, plugin tools, permissions, memory, limits, and
      expected artifacts.
- [ ] Let the user edit, approve, reject, save as a template, or launch the
      proposal; require fresh approval when a consequential capability changes.
- [ ] Launch through the existing supervisor/backend contract and connect the
      resulting run, approvals, tools, and artifacts to the activity timeline.
- [ ] Add coding, research, email-draft, and presentation proposal fixtures
      using mock capabilities.
- [ ] Add tests proving prompt text alone cannot smuggle unlisted tools or
      exceed the approved workspace/capabilities.

## Definition of Done

- The orchestrator proposes a schema-valid blueprint from an outcome request
  (test: `test_orchestrator_proposes_valid_agent_blueprint`).
- Nothing launches until the reviewed blueprint is explicitly approved
  (test: `orchestrator-blueprint-approval.spec.ts`).
- The launched agent receives exactly the approved effective capabilities
  (test: `test_blueprint_launch_uses_exact_approved_grants`).
- Changing tools, workspace, or outward permissions invalidates prior approval
  (test: `test_material_blueprint_change_requires_reapproval`).
- manual: specialized agent creation is understandable without reading raw JSON
  or backend CLI flags.

## Notes

- Epic: 20260729-102204.
- Depends on: 20260729-102206, 20260729-102207, 20260729-102919, and
  20260729-102203.
- Keep proposal generation separate from approval and enforcement.

## Flow State

- FLOW STEP: PLANNING
