# Retro: Projects backend - first-class Project store + CRUD API

- TASK: 20260720-210644
- BRANCH: feature/projects-store
- REVIEW ROUNDS: 1 (APPROVE, 2 NITs fixed in-round)

## What went well

- The SettingsStore from the settings-console goal was a ready template:
  atomic-persist, tolerant-load, and the read-only gate transcribed directly,
  so this was mostly the slug/cwd-validation specifics. Compounding paying off.
- Confined the id to a URL-safe slug up front (fullmatch, provable charset), so
  the reviewer's traversal/injection probes all came back clean.

## What went wrong

- Two NITs, both minor: endpoint-level read-only gate was only asserted for
  POST (not PATCH/DELETE), and `_slugify`'s non-ASCII dropping was undocumented.
  Both were the reviewer catching thin test coverage / missing intent-comment,
  not a real defect.

## What to improve next time

- When several endpoints share a gate (read-only), assert it at the HTTP layer
  for EACH verb, not just one - the same shape as the T5 "test the net-new
  route" lesson.

## Action items

- No lessons ledger entry (covered by the existing net-new-route lesson).
- No follow-up code task. PB (tatr-tasks endpoint) builds on this.
