# Review: Settings backend - named config profiles

- TASK: 20260720-184138
- BRANCH: feature/settings-profiles

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (fresh subagent, ran the full nix suite; in-session
  pass re-ran the suite and adopted both NIT fixes)

No BLOCKER/MAJOR. Full suite green (ruff + mypy + pytest via `python -m pytest`,
176 tests). Reviewer verified the reset-to-base crux (`_base_values` captured
before `_load`; `activate` resets then applies; an unset key falls back to env
default), profile-name fullmatch validation, both delete guards (active + last),
live on_change only on a rebuild-class change, persistence across a fresh store,
and the read-only gate on every profile op.

- [x] R1.1 (NIT) settings_store.py create_profile - shallow `dict(...)` copy
  shared nested list objects with the source profile (latent, not exploitable
  since apply/activate always reassign).
  - Response: fixed - `copy.deepcopy(self._overrides())` so profiles never
    share nested lists.
- [x] R1.2 (NIT) DoD proof name drift (`cannot_delete_active_profile` shipped as
  `test_cannot_delete_active_or_last_profile`).
  - Response: noted the rename in TASK.md Close-out; the shipped test covers the
    DoD case and the last-profile case.

No open `manual:` DoD items.
