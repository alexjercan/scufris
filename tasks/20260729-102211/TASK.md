# Spike: define the artifact and viewer extension model

- STATUS: OPEN
- PRIORITY: 0
- TAGS: spike,backlog,artifacts,plugins,frontend

## Story

As a platform designer, I want one artifact and viewer extension model, so that
agent outputs, task records, research sources, drafts, PDFs, and presentations
share provenance and access controls instead of each feature inventing a file
path convention.

## Steps

- [ ] Inventory current chat attachments, exports, task records, run outcomes,
      workspace paths, and likely plugin-generated outputs.
- [ ] Define artifact identity, owner/project/run, media type, source/provenance,
      storage or external reference, checksum, size, version, preview, citation,
      retention, and capability metadata.
- [ ] Compare managed blob storage, workspace-relative references, and external
      provider references, including stale/moved files and large documents.
- [ ] Define viewer contribution registration without allowing untrusted
      arbitrary frontend code in the initial version.
- [ ] Define read, download/export, edit, version, diff, and save authorization
      boundaries plus path/symlink and content-size handling.
- [ ] Define renderer sandboxing and sanitization requirements per supported
      content type.
- [ ] Write `SPIKE.md`, record the chosen model in `DECISION.md`, and refine the
      remaining artifact children.

## Definition of Done

- The spike includes example records for Markdown, image, PDF, email draft,
  calendar event, and PPTX artifacts
  (cmd: `rg -n "Markdown|image|PDF|email|calendar|PPTX" tasks/20260729-102211/SPIKE.md`).
- Storage/reference, viewer extension, provenance, and authorization choices
  are recorded
  (cmd: `test -f tasks/20260729-102211/SPIKE.md && test -f tasks/20260729-102211/DECISION.md && tatr check 20260729-102211`).
- Threat cases cover path traversal, symlink escape, active content, oversized
  input, stale references, and secret leakage
  (cmd: `rg -n "traversal|symlink|active content|oversized|stale|secret" tasks/20260729-102211/SPIKE.md`).
- manual: the user accepts what Scufris manages versus merely references.

## Notes

- Epic: 20260729-102210.
- Depends on the blueprint/plugin direction in 20260729-102205.
- The project task artifact viewer can inform this design, but task records
  remain a more constrained read-only source.

## Flow State

- FLOW STEP: PLANNING
