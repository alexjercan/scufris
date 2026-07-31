# Condense repository agent guidance - notes

## Changed

- Reduced `AGENTS.md` from 620 lines to 142 lines.
- Kept commands, workflow, testing, implementation rules, and security invariants.
- Replaced duplicated architecture prose with authoritative README and decision pointers.
- Moved the release runbook to `docs/RELEASING.md`.
- Updated both root README release pointers.

## Why

- Critical rules were buried inside duplicated architecture and release prose.
- Duplicated detail could drift from live package READMEs and workflows.
- Short invariants keep unsafe changes visible without restating their full rationale.

## Tradeoffs

- Cold readers follow links for design rationale and protocol detail.
- `AGENTS.md` remains longer than a generic repository guide because host mutation boundaries are safety-critical.

## Review fixes

- Scoped operator-session approval to HTTP; preserved Telegram's allowlisted credential path.
- Pointed the lower root README release link directly to `docs/RELEASING.md`.
- Documented `web/src/login.ts` as the bare-fetch bootstrap exception.

## Verification

- Five required `Agent workflow` pointers: pass.
- Release document and both README pointers: pass.
- ASCII punctuation and `git diff --check`: pass.
- `tatr check 20260731-131543`: pass before review records.
- Full ledger check: pre-existing failures requiring user promotion dispositions.

## Next time

- Grep every live README pointer after moving a section.
- Qualify security summaries by transport; HTTP and Telegram use different operator credentials.
