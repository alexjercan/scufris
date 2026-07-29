# Add agent blueprint schemas and reusable templates

- STATUS: OPEN
- PRIORITY: 0
- TAGS: feature,backlog,agents,plugins,backend

## Story

As an operator, I want reusable agent blueprints and templates, so that I can
create coding, research, writing, and automation specialists consistently
without re-entering low-level backend configuration each time.

## Steps

- [ ] Implement the versioned Pydantic blueprint schema selected by
      20260729-102205, with validation and readable validation errors.
- [ ] Support named reusable templates, project defaults, controlled overrides,
      and a fully resolved effective-blueprint view.
- [ ] Migrate existing `AgentRecord` data into the new model without losing
      current agent behavior or reserved-orchestrator semantics.
- [ ] Validate harness/model compatibility, workspace roots, referenced skills,
      plugins, capabilities, memory policy, limits, and artifact expectations.
- [ ] Add CRUD and resolve/preview API endpoints without exposing secret values.
- [ ] Add fixtures for coding, research, email-draft, and presentation
      blueprints as examples, not privileged built-ins.
- [ ] Add backend contract tests proving each current harness receives the
      resolved subset it supports and rejects unsupported requirements.

## Definition of Done

- Blueprints serialize, validate, migrate, resolve inheritance, and preserve
  existing agents (test: `test_agent_blueprint_roundtrip_and_migration`).
- Invalid harness/capability/workspace combinations fail before launch
  (test: `test_agent_blueprint_rejects_incompatible_requirements`).
- API responses expose secret references but never secret values
  (test: `test_agent_blueprint_api_redacts_secrets`).
- Example templates instantiate under the mock backend
  (cmd: `python examples/agent_blueprints.py`).

## Notes

- Epic: 20260729-102204.
- Depends on: 20260729-102205 and 20260729-102147.
- Keep blueprint storage separate from one harness's CLI argument model.

## Flow State

- FLOW STEP: PLANNING
