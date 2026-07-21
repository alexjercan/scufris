# Review: F6 per-backend model autocomplete (datalist)

- TASK: 20260721-152737
- BRANCH: feature/model-dropdown

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Both suites ran green in the worktree (ruff + mypy 35 files + pytest; web
npm run ci, 166 tests). Zero findings.

Verified by the reviewer:
- `models_for` returns the catalog with the configured default prepended only
  when outside it (no dup), via `canonical_backend`; empty catalog -> [].
  `BackendOption.models` populated in the endpoint.
- The model input's `list` references its datalist; `fillModelList` populates
  from the backend's models; a backend change swaps options AND re-defaults the
  value (MB1 preserved); free text round-trips. `modelList` is appended by both
  callers so the `list` resolves in the DOM; the two datalist ids differ
  ("new agent" vs "agent settings") and render on separate pages - no collision.
- New tests genuinely gate each behavior (exact option arrays, free-text
  survival, endpoint catalog + env-override-prepend). All three frontend
  BackendOption fixtures updated with `models`; no fixture builds one without it.
- Close-out matches the code.

No BLOCKER/MAJOR/MINOR/NIT issues. APPROVE.
