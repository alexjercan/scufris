# Settings backend: named config profiles

- STATUS: OPEN
- PRIORITY: 38
- TAGS: feature,agent,backend,config

## Story

As the operator, I want named config PROFILES I can save and switch between
(e.g. "default", "offline-mock", "cheap-model"), so I can flip a whole set of
settings at once. This is single-operator convenience, NOT multi-user auth.

## Steps

- [ ] Extend the task-1 store's on-disk shape to named profiles:
      `{active: <name>, profiles: {<name>: {<overrides>}}}` (the shape task 1
      already reserved). Effective settings = env base <- active profile's
      overrides.
- [ ] Add profile endpoints: `GET /api/agent/profiles` (list names + active),
      `POST /api/agent/profiles` (create/rename/save-current),
      `POST /api/agent/profiles/activate {name}` (switch active),
      `DELETE /api/agent/profiles/{name}` (cannot delete the active/last one).
- [ ] Switching the active profile changes effective config for subsequent
      requests without a restart (reuse the task-1 live provider).
- [ ] Tests: save a profile, switch to it, effective config reflects its
      overrides; switch back; delete a non-active profile; deleting the active
      or last profile is refused.

## Definition of Done

- Saving and activating a profile changes the effective config; switching away
  restores the other profile's config (test: `profile_switch_changes_config`).
- The active profile persists across a restart (test: `active_profile_persists`).
- Deleting the active or last profile is refused with a clear error
  (test: `cannot_delete_active_profile`).
- Full suite green (cmd: `nix develop --command bash -c "ruff check . && mypy . && pytest -q"`).

## Notes

- Depends on: 20260720-184136 (store + live provider).
- Keep it file-based (the store's JSON), consistent with the repo's ethos - no
  DB. A profile holds only whitelisted override keys.
- "per user configs" from the goal == these named profiles for the single
  operator; do NOT add users/auth.
