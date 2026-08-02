# Add Markdown text diff image artifact viewers

- PRIORITY: 0
- TAGS: feature, backlog, artifacts, frontend
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Story

As an agent user, I want Markdown, plain text, code/diffs, and images rendered
directly in Scufris, so that common outputs are inspectable without switching
applications or trusting active document content.

## Steps

- [ ] Implement the artifact registry/read APIs selected by 20260729-102211
      with project/run scoping, range/size limits, and safe media-type handling.
- [ ] Build viewer routing and shared metadata for title, type, size, checksum,
      provenance, producing agent/run, versions, and export action.
- [ ] Add safe Markdown, plain-text/code, unified-diff, and image viewers with
      syntax, line, zoom, wrap, and side-by-side controls where appropriate.
- [ ] Reject active HTML/SVG/script behavior, unsafe links, unsupported
      encodings, decompression bombs, traversal, and symlink escape.
- [ ] Add responsive, keyboard, screen-reader, loading, missing, stale, and
      large-artifact states.
- [ ] Link artifacts from chat messages and the run activity timeline.

## Definition of Done

- All four viewer types render from a real agent-run fixture
  (test: `artifact-viewers.spec.ts`).
- Hostile Markdown/image/path fixtures cannot execute content or escape the
  allowed artifact boundary (test: `test_artifact_viewer_hostile_inputs`).
- Large text and image fixtures remain bounded and usable
  (test: `artifact-viewers-large.spec.ts`).
- Artifact links survive reload and retain run/project context
  (test: `artifact-viewer-history.spec.ts`).

## Notes

- Epic: 20260729-102210.
- Depends on: 20260729-102211, 20260729-102147, 20260729-102152, and
  20260729-102203.
- Use established libraries for syntax highlighting and image handling where
  they materially reduce parser/rendering risk.
