# Add explicit diff save approval for artifact editing

- PRIORITY: 0
- TAGS: feature, backlog, artifacts, security, frontend
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Story

As an operator, I want agent-proposed file edits shown as exact versioned diffs
before saving, so that I can use Scufris for writing and document automation
without hidden or stale writes.

## Steps

- [ ] Define an edit proposal containing base checksum/version, target artifact,
      replacement or patch, producing run, rationale, and required capability.
- [ ] Add preview APIs and diff rendering without modifying the source file.
- [ ] Require explicit approval through the capability system before atomic
      save, and reject stale base versions instead of overwriting newer work.
- [ ] Create a new artifact version for every accepted edit and support viewing
      history and reverting through another explicit proposal.
- [ ] Enforce project roots, file-type allowlists, symlink/path boundaries,
      size limits, and read-only/untrusted artifact restrictions.
- [ ] Add Markdown/text editing first; require format-specific validators before
      enabling PDF, PPTX, or other structured-format writes.
- [ ] Add concurrent-edit, stale-base, rejected-approval, failed-write,
      permission-revocation, and restart tests.

## Definition of Done

- Previewing or rejecting an edit cannot alter the source
  (test: `test_artifact_edit_preview_is_side_effect_free`).
- Approved writes are atomic, versioned, audited, and tied to the exact diff
  shown to the user (test: `artifact-edit-approval.spec.ts`).
- Stale or concurrently changed files are never overwritten
  (test: `test_artifact_edit_rejects_stale_base`).
- Path, symlink, and capability escapes are rejected
  (test: `test_artifact_edit_enforces_write_boundary`).

## Notes

- Epic: 20260729-102210.
- Depends on: 20260729-102212 and 20260729-102919.
- Editing is a capability-controlled workflow, not an unrestricted web IDE.
