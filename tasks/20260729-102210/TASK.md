# EPIC: Add rich artifacts and personal automation plugins

- PRIORITY: 0
- TAGS: goal, epic, backlog, artifacts, plugins
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Epic

Give agents durable, inspectable outputs and extend Scufris into useful personal
automation without turning the core app into a full office suite. Start with a
safe artifact model and read-only viewers, then add explicit diff-and-approval
editing and narrowly scoped email, calendar, PDF, and presentation plugins.

## Done Means

1. Agents and plugins can register typed, provenance-bearing artifacts that are
   safely viewed without arbitrary filesystem access
   (test: `test_artifact_registration_and_access_boundary`).
2. Markdown, text, diff, image, and PDF outputs render read-only in the browser
   with source and citation context (test: `artifact-viewers.spec.ts`).
3. File changes require an explicit previewed diff and approval before atomic
   save (test: `artifact-edit-approval.spec.ts`).
4. Email/calendar and PPTX examples run through the plugin/capability model
   rather than bespoke core integrations
   (cmd: `python examples/personal_information_plugins.py && python examples/presentation_agent.py`).
5. manual: generated documents are easy to inspect, validate, and associate
   with the agent run that produced them.

## Child Tasks

- [ ] 20260729-102211 (p0, scufris) define the artifact and viewer extension
      model
- [ ] 20260729-102212 (p0, scufris) add Markdown, text, diff, and image artifact
      viewers
- [ ] 20260729-102214 (p0, scufris) add PDF preview, extraction, and source
      citations
- [ ] 20260729-102215 (p0, scufris) add explicit diff/save approval for artifact
      editing
- [ ] 20260729-102216 (p0, scufris) add read-only email search and calendar
      agenda plugins
- [ ] 20260729-102217 (p0, scufris) add PPTX generation, preview, and validation
      plugin

## Decisions

- Pending 20260729-102211 SPIKE.md and DECISION.md: artifact ownership,
  storage/reference model, viewer registration, and safe file boundary.

## Manual Acceptance

- (pending) 20260729-102214: PDF previews and extracted citations are useful
  for real research material.
- (pending) 20260729-102215: edit approval makes the exact write obvious.
- (pending) 20260729-102217: a generated presentation is visually reviewable
  before export.

## Deferred (2026-07-29 backlog review)

This epic mixes a foundation with demos, and only the foundation is clearly
wanted. The artifact model and read-only viewers (20260729-102211,
20260729-102212, 20260729-102215) stay: every agent run should produce
inspectable, attributable output, and that is worth building on its own terms.

The personal-automation plugins are parked until a concrete need appears:

- 20260729-102217 (PPTX generation): parked. Nothing in current use makes
  presentations; build it when a real deck is needed, not to prove the plugin
  model.
- 20260729-102216 (email and calendar): parked. If the plugin model needs a
  proof, ONE read-only integration is enough - pick it when the plugin epic is
  scheduled, do not build both.
- 20260729-102214 (PDF preview and extraction): keep, but it is only worth it
  alongside real research material - it is coupled to the research epic
  (20260729-102218), not to this one.

## Sequencing

- Post-v0.1.0 order (2026-07-29 backlog review): FOURTH of the five backlog
  epics, foundation only (see Deferred above). The artifact model becomes
  clearly worth it once several agent kinds produce output worth keeping - by
  then the host operator epic will have shipped previews, diffs, and audit
  records that want a home.
- Stays `backlog` at priority 0 until pulled into a release plan.
