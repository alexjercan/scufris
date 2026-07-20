# Review: Projects backend - first-class Project store + CRUD API

- TASK: 20260720-210644
- BRANCH: feature/projects-store

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (fresh subagent, ran the full nix suite; in-session
  pass adopted both NIT fixes and re-ran)

No BLOCKER/MAJOR/MINOR. Full suite green (ruff + mypy + pytest via `python -m
pytest`, 198 tests). Reviewer verified: atomic persist (temp + os.replace),
tolerant load (missing/corrupt/non-list handled, invalid records dropped),
round-trip through a fresh store; `_slugify` output provably confined to
`[A-Za-z0-9-]` (tested against `../etc`, `a/b`, `..`, `!!!` - no traversal),
PROJECT_ID_RE via `re.fullmatch`; cwd-existence + empty-name validation ->
422; 404 on unknown; 403 read-only gate on create/update/delete. All four
DoD-named tests present and fail if their mechanism is removed.

- [x] R1.1 (NIT) endpoint-level read-only gate only asserted for POST.
  - Response: fixed - added `client.patch(...)`/`client.delete(...)` 403
    assertions to `test_projects_write_forbidden_when_readonly`.
- [x] R1.2 (NIT) `_slugify` drops non-ASCII (lossy, still correct).
  - Response: fixed - added a docstring noting transliteration is intentionally
    not done and the charset confinement is the point.

No open `manual:` DoD items.
