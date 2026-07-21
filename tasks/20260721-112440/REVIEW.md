# Review: sesh.py directory discovery + Projects discovery/create (no tmux)

- TASK: 20260721-112440
- BRANCH: feature/projects-discovery

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (round-1 findings from a fresh subagent with no sight
  of the implementing session; the in-session pass re-ran both suites and adopted
  both findings, re-deriving the traversal/allowed-base security check)

Both check suites pass: backend `ruff` + `mypy` + `pytest` (all green) and web
`npm run ci` (prettier + eslint + vitest + webpack build). The diff delivers the
Goal: `sesh.discover` scans one level deep (files/hidden skipped, deduped by
resolved path, missing base ignored) and infers language from marker files;
`sesh.create` mkdirs under a base with NO subprocess and rejects any non-single-
segment name; `GET /api/projects/discovered` unions discovered + registered (each
flagged) + base dirs; `POST /api/projects/new` mkdirs only under an allowed base
(422 otherwise, 403 read-only before any mkdir) then registers. Security surface
(traversal, allowed-base resolve-matching, route ordering, read-only-before-mkdir,
frontend escaping) verified with no way to mkdir outside the allowlist.

- [x] R1.1 (MINOR) `.env.example` - the new `SCUFRIS_PROJECT_BASE_DIRS` knob was
  not documented alongside the ~20 other `SCUFRIS_` settings. Add a commented
  example line.
  - Response: Fixed. Added a commented `# SCUFRIS_PROJECT_BASE_DIRS=~/personal:~/work`
    line (with a one-line explanation + the default set) to `.env.example`.

- [x] R1.2 (NIT) tests/test_sesh.py - `create`'s rejection tests covered `/`, `\`,
  `..`, `.`, NUL and blank names but not an absolute-path name; add one for
  completeness (it is already rejected by the leading-`/` fail).
  - Response: Fixed. Added `/etc/foo` to the rejected-names list in
    `test_create_rejects_traversal_and_separators`. Confirmed green.

### Pending manual DoD (user's to eyeball; APPROVE does not resolve it)

- manual: the Projects page lists my real dirs and creating one works end to end.
