# Add PPTX generation preview and validation plugin

- STATUS: OPEN
- PRIORITY: 0
- TAGS: feature,backlog,plugins,pptx,artifacts

## Story

As a project user, I want a presentation specialist to generate, preview, and
validate PPTX files, so that Scufris can automate decks while keeping the
result inspectable and editable outside the app.

## Steps

- [ ] Select a maintained PPTX generation library and headless render/validation
      pipeline that packages reproducibly under Nix.
- [ ] Define a presentation artifact contract for source outline/data, theme,
      generated PPTX, rendered slide previews, validation report, and
      provenance.
- [ ] Implement an out-of-process plugin with typed tools for create deck,
      update slide content, render preview, and validate structure/layout.
- [ ] Add a presentation agent template that produces an outline for approval
      before generating the file and exposes all input artifacts.
- [ ] Render every slide to a browser-viewable preview and detect missing fonts,
      clipped/overflowing text, broken media, empty slides, and invalid files.
- [ ] Add deterministic fixtures for charts, tables, images, speaker notes,
      long text, missing media, theme fonts, and regeneration.
- [ ] Add a runnable example that creates and validates a small project deck.

## Definition of Done

- The example produces a valid PPTX, slide previews, and validation report
  (cmd: `python examples/presentation_agent.py`).
- Generated slide previews are visible and linked to the producing run
  (test: `pptx-artifact-viewer.spec.ts`).
- Layout overflow, missing media/fonts, and corrupt packages fail validation
  with slide-level diagnostics (test: `test_pptx_validation_failures`).
- Re-running from the same approved inputs is deterministic
  (test: `test_pptx_generation_is_reproducible`).
- manual: a generated real-project deck is usable after normal human editing.

## Notes

- Epic: 20260729-102210.
- Depends on: 20260729-102207, 20260729-102212, and 20260729-102919.
- Record the generation and rendering toolchain in `DECISION.md`.
- Do not build a full in-browser PowerPoint editor in this task.

## Flow State

- FLOW STEP: PLANNING
