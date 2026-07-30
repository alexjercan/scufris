# Settings backend: named config profiles

- STATUS: CLOSED
- PRIORITY: 38
- TAGS: feature,agent,backend,config
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

As the operator, I want named config PROFILES I can save and switch between
(e.g. "default", "offline-mock", "cheap-model"), so I can flip a whole set of
settings at once. This is single-operator convenience, NOT multi-user auth.

## Steps

- [x] Extend the store to named profiles on the reserved
      `{active, profiles: {<name>: {<overrides>}}}` shape. Effective settings =
      env base <- active profile's overrides. Snapshot the env-base values at
      init so `activate` can RESET to base then apply the target profile (a key
      the target does not override falls back to env default, not the old
      profile's value).
- [x] Profile endpoints: `GET /api/agent/profiles` (names + active),
      `POST /api/agent/profiles` (create, `copy_from_active` default true),
      `POST /api/agent/profiles/activate {name}` (switch -> returns effective
      config), `DELETE /api/agent/profiles/{name}` (refuses active/last).
      (Amended: "rename/save-current" is expressed as create-with-copy; a bare
      rename was not needed for the UI and would complicate the active pointer.)
- [x] Switching is live: `activate` resets+reapplies onto the in-place settings
      and fires `on_change` for a rebuild-class key (agent rebuilt via T1's
      AgentHandle).
- [x] Tests: store-level (switch changes config, empty profile falls back to
      base, active persists across restart, cannot delete active/last, dup/bad
      name, unknown, read-only gate, on_change on activate) + endpoint flow
      (list/create/activate/delete + guards + 403).

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

## Close-out

- The one non-obvious piece: because T1 mutates the single Settings object in
  place, switching profiles cannot just "apply the new overrides" - the object
  still carries the OLD profile's mutations. So the store snapshots each
  writable key's env-base value at init (`_base_values`, captured BEFORE
  `_load`), and `activate` does reset-to-base then apply-target. Test
  `test_activate_resets_keys_the_target_does_not_override` pins exactly this
  (an empty profile falls back to env base, not the previous profile's value).
- `activate` reuses T1's live path: it fires `on_change` only when a
  rebuild-class key (agent_enabled/agent_backend) actually differs after the
  swap, so switching to a profile with a different backend rebuilds the agent.
- Profile names are validated with `PROFILE_NAME_RE` (fullmatch) because the
  name is both a JSON key and a URL path segment (`DELETE .../profiles/{name}`)
  - no slashes/dots/spaces.
- "delete active or last" is really one condition (the active is always one of
  the set), but both checks are kept explicit to match the DoD and give a clear
  message either way.
- DoD proof rename: `cannot_delete_active_profile` shipped as
  `test_cannot_delete_active_or_last_profile` (covers both guards). Review R1.2.
- Review R1.1: `create_profile` now `copy.deepcopy`s the source overrides so a
  new profile never shares nested list objects (latent aliasing footgun,
  defensive fix).
- Self-reflection: smooth task - T1's profile-ready on-disk shape and live
  provider meant this was almost purely additive. The base-snapshot need was
  spotted by reasoning about the in-place mutation before coding, avoiding a
  "switch leaves stale values" bug.
