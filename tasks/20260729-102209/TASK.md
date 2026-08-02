# Let the orchestrator propose approve and launch specialist agents

- PRIORITY: 0
- TAGS: feature, backlog, agents, orchestrator, frontend
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Story

As a Scufris user, I want to describe an outcome and receive a transparent,
editable specialist-agent proposal, so that the orchestrator can resolve a
reusable preset into the right harness, project/task assignment, tools,
permissions, and output contract without silently enabling capabilities.

## Steps

- [ ] Add a structured orchestrator tool that proposes an agent instance and
      run assignment from the available presets, backends, project/task
      context, stage guards, and policies.
- [ ] Validate proposals server-side and explain missing, incompatible,
      unhealthy, ungranted, or workflow-illegal requirements.
- [ ] Build a review surface showing name/purpose, harness/model, project root,
      task/stage assignment, instructions/skills, built-in and optional plugin
      tools, permissions, session/memory policy, limits, and expected artifacts.
- [ ] Let the user edit, approve, reject, save as a template, or launch the
      proposal; require fresh approval when a consequential capability changes.
- [ ] Launch through the transport-independent service, supervisor, and backend
      contracts and connect the resulting assignment, run, native session,
      approvals, tools, artifacts, activity, and semantic conversation report.
- [ ] Add planning, implementation, review, host, and optional plugin-backed
      proposal fixtures using the mock backend and capabilities.
- [ ] Add tests proving prompt text alone cannot smuggle unlisted tools or
      exceed the approved workspace, workflow stage, or capabilities.

## Definition of Done

- The orchestrator proposes a schema-valid preset-derived agent and assignment
  from an outcome request
  (test: `test_orchestrator_proposes_valid_specialist_agent`).
- Nothing launches until the reviewed proposal is explicitly approved
  (test: `orchestrator-agent-approval.spec.ts`).
- The launched agent receives exactly the approved effective capabilities
  (test: `test_agent_launch_uses_exact_approved_requirements`).
- Changing tools, workspace, or outward permissions invalidates prior approval
  (test: `test_material_agent_proposal_change_requires_reapproval`).
- Specialized agent creation is understandable without reading raw JSON
  or backend CLI flags (manual: user check).

## Notes

- Epic: 20260729-102204.
- Base project-agent launch depends on: 20260729-220835 and the implementation
  tasks it seeds, 20260729-103712, 20260729-102158, 20260729-102203, and
  20260729-102206.
- The base planning/work/review path must not depend on the general plugin
  registry or capability-grant platform. A proposal that requests optional
  plugin capabilities additionally depends on 20260729-102207,
  20260729-102208, and 20260729-102919 and must use their enforcement gates.
- Adopt this task into the future actor-aware orchestrator epic after
  20260729-220835 is accepted; refine its exact boundary against that epic
  rather than creating a competing conversation or workflow model here.
- Keep proposal generation separate from approval and enforcement.
