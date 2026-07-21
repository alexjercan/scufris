# Retro: B1 backend surface cleanup

- TASK: 20260721-112429
- BRANCH: refactor/backend-surface (landed ba8203c)
- REVIEW ROUNDS: 1 out-of-context APPROVE (1 MINOR tracked, 3 NIT) + in-session round 2

## What went well

- Splitting "resolve always / gate on create" cleanly solved the mock-flag
  design: an existing mock agent keeps running even if the flag is later turned
  off, because only creation is gated. The reviewer confirmed this holds.
- Putting `canonical_backend`/`available_backends`/`default_model_for` in config
  (not backends) kept `agent_store` from importing the heavy backend runners -
  respecting the A5 "keep MCP-server imports light" lesson without being reminded.
- Legacy normalize-on-load + persist-on-next-write migrates old records with no
  migration script and no dropped data.

## What went wrong

- Nothing significant. The one loose end is a SEPARATE field: the settings page
  still shows raw `app_server`/`exec`/`mock` for the process chat agent's
  `agent_backend`. Two backend vocabularies now coexist. Not a B1 bug (out of
  scope), but a real inconsistency - tracked on B5 rather than left to rot.

## What to improve next time

- When cleaning a "vocabulary" (backend ids), grep for EVERY surface that uses
  the old vocabulary up front (found the settings picker only via review). The
  process `agent_backend` field and the per-agent `backend` field look alike but
  are different - a vocabulary change should audit both at plan time.

## Action items

- [x] MINOR (settings picker) tracked as a carried-in note on B5.
- [x] NIT (redundant paren) fixed.
- No new ledger entry: the "audit all surfaces of a vocabulary" point is captured
  here and actioned on B5.
