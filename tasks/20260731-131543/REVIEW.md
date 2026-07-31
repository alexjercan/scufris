# Review: Condense repository agent guidance

- TASK: 20260731-131543
- BRANCH: master (uncommitted working-tree diff)

## Round 1

- REVIEWER: out-of-context
- VERDICT: APPROVE

- [x] R1.1 (MAJOR) AGENTS.md:107 - "operator session only" contradicted the supported allowlisted Telegram approval surface. Scope sessions to HTTP decisions and state that Telegram derives operator identity from its allowlisted chat credential.
  - Response: Scoped HTTP decisions to sessions and documented Telegram's allowlisted chat credential. Reviewer re-verified.
- [x] R1.2 (MINOR) README.md:393 - the release-procedure pointer still linked to `AGENTS.md#releasing`. Link directly to `docs/RELEASING.md`.
  - Response: Updated the direct link. Reviewer re-verified.

Verified: diff against HEAD, retained-rule comparison, five workflow pointers,
new release document and pointers, referenced paths, ASCII punctuation, and
`git diff --check`.

Not verified: full ledger conformance. `tatr check --ledger LESSONS.md` fails
on pre-existing promotion decisions that require user disposition.

Pending user check: compare retained rules and pointers against the previous
`AGENTS.md`.
