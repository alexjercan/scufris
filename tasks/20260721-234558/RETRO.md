# Retro: U1 - orchestrator as a first-class hidden, editable agent

- TASK: 20260721-234558
- BRANCH: feature/orchestrator-first-class (landed 10c54d3)
- REVIEW ROUNDS: 1 (APPROVE, out-of-context, 2 NITs no-action)

See TASK.md for what/why and REVIEW.md for the findings. Process notes only here.

## What went well

- Recon-first pinned the exact crux before any edit: `list()` prepends the
  orchestrator; `update()` 409s it; its config maps to settings keys. The plan
  named all three, so the implementation was mechanical.
- Kept the guard invariant instead of weakening it: adding `claude_model` +
  `agent_permission_mode` to WRITABLE_KEYS *and* AgentConfigUpdate kept the
  `test_writable_keys_match_the_api_update_model` equality green - a deliberate
  choice over relaxing the assertion.
- Grabbed the cheap in-scope doc win: the `.env.example` backend doc was still
  the pre-B5e `app_server` vocabulary; refreshed it to codex|claude|mock while
  adding the new knob (the `new-config-field-updates-all-its-surfaces` lesson).

## What went wrong

- Nothing beyond the expected: the first full backend run showed 5 breakages, ALL
  the same shape - assertions that the synthetic orchestrator is present-in/first-of
  the list, plus the whitelist-sync test. Not a mistake (they were anticipated in
  the plan), but a clean confirmation that the synthetic-item lesson cuts BOTH
  ways: adding a synthetic broke "empty" assertions (last EPIC); removing it from
  the list breaks the "is present" assertions and re-enables the empty state.

## What to improve next time

- When hiding/removing an always-present synthetic member, grep for the "is
  present / is first / len == N" assertions up front (the mirror of the add case)
  and flip them in the same pass - which I did, but worth making the reflex
  explicit.

## Action items

- [x] Bumped `always-present-synthetic-item-invalidates-empty-assertions` to x2
      (now covers add AND remove of the synthetic).
- No follow-up tasks; U2 (per-agent panel data) is next in the flow.
