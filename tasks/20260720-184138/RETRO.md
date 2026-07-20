# Retro: Settings backend - named config profiles

- TASK: 20260720-184138
- BRANCH: feature/settings-profiles
- REVIEW ROUNDS: 1 (APPROVE; 2 NITs fixed in-round)

## What went well

- Almost purely additive thanks to T1: the store's on-disk shape was already
  `{active, profiles:{...}}` and the live-provider path existed, so profiles
  slotted in without a rewrite. Deliberately reserving that shape in T1 paid
  off exactly as intended.
- Caught the reset-to-base subtlety by reasoning about the in-place mutation
  BEFORE coding: because the settings object carries the old profile's
  mutations, switching must reset to a captured env-base snapshot then apply
  the target. Pinned it with a dedicated test (empty profile -> env default,
  not the previous profile's value).

## What went wrong

- Two NITs from review, both minor: a shallow copy that shared nested list
  objects between profiles (latent, not exploitable because every path
  reassigns rather than mutates) - fixed with deepcopy; and a DoD proof
  name-drift (`cannot_delete_active_profile` vs the shipped
  `test_cannot_delete_active_or_last_profile`).

## What to improve next time

- Keep the DoD's named proof and the shipped test name identical, or note the
  rename in Close-out at write time - the reviewer flagged the drift because a
  literal grep for the DoD name fails. (Recurring across T1-T3; worth a habit.)
- Default to `copy.deepcopy` when snapshotting a nested-mutable dict that will
  live independently, even when current paths only reassign.

## Action items

- No new lessons ledger entry (the reset-to-base insight is task-specific; the
  deepcopy/DoD-name points are minor habits, captured here).
- No follow-up code task.
