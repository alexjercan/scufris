# Add reusable agent preset schemas and templates

- STATUS: OPEN
- PRIORITY: 61
- TAGS: feature, v0.2.0, agents, backend

## Story

As an operator, I want reusable agent presets and templates, so that I can
create coding, research, writing, and automation specialists consistently
without re-entering low-level backend configuration each time.

## Steps

- [ ] Implement the Pydantic preset schema selected by
      20260729-102205, with validation and readable validation errors.
- [ ] Support named reusable templates, project defaults, controlled overrides,
      stage defaults, and a fully resolved effective-preset view.
- [ ] Preserve current `AgentRecord` behavior and reserved-orchestrator
      semantics through an explicit compatibility adapter; do not force every
      existing agent to become a new record shape in one migration.
- [ ] Validate backend/model compatibility, workspace roots, referenced skills
      and known capability/tool IDs, memory/session policy, limits, and output
      expectations before launch.
- [ ] Add CRUD and resolve/preview API endpoints; responses show references and
      effective values but never resolve future secret values.
- [ ] Add concrete planning, implementation, review, and host presets as
      examples, not privileged built-ins. Allow project policy to select a
      different preset/backend per stage.
- [ ] Add backend contract tests proving each current harness receives the
      resolved subset it supports and rejects unsupported requirements.

## Definition of Done

- Presets serialize, validate, resolve project/stage defaults and overrides, and
  preserve existing agents through the compatibility adapter
  (test: `test_agent_preset_roundtrip_and_compatibility`).
- Invalid backend/capability/workspace combinations fail before launch
  (test: `test_agent_preset_rejects_incompatible_requirements`).
- A project resolves distinct plan/work/review presets and their effective
  backends without mutating the reusable templates
  (test: `test_project_stage_presets_resolve_independently`).
- API responses expose capability/secret references but never secret values
  (test: `test_agent_preset_api_never_resolves_secret_values`).
- Example templates instantiate under the mock backend
  (cmd: `python examples/agent_presets.py`).

## Notes

- Epic: 20260729-102204.
- Depends on: 20260729-102205, 20260729-102147, and 20260729-220835.
- V0.2.0 readiness scope stops at preset storage, resolution, validation,
  preview, examples, and backend mapping. Orchestrator proposal/approval/launch
  remains 20260729-102209.
- Keep preset storage separate from one harness's CLI argument model.

## Flow State

- FLOW STEP: PLANNING
