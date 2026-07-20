# Retro: A1 AgentStore (agent as a first-class record + CRUD)

- TASK: 20260720-221929
- BRANCH: feature/agent-store (landed 17bad00)
- REVIEW ROUNDS: 1 out-of-context APPROVE (1 MINOR + 2 NIT, all addressed) + in-session round 2 confirming the cleanups

## What went well

- Mirroring `projects.py` made this fast and low-risk: same persistence shape,
  same gate ordering, same test structure. Reading the template first meant the
  only real design decision (FK-vs-cwd-snapshot for the project link) was made
  deliberately, toward the FK, keeping Project the single source of cwd.
- The `test-the-net-new-route-not-the-reused-path` lesson was applied up front:
  every new route got its own 403/404/422 test rather than leaning on the
  project routes' coverage.

## What went wrong

- Copied a vestigial `DuplicateAgent` exception from the template's
  `DuplicateProject`, which is itself unreachable (the `_unique_id` dedup means a
  create never collides). Harmless but dead. Root cause: mirroring a template
  wholesale carries its dead code along with its good parts. The reviewer caught
  it; removed.

## What to improve next time

- When mirroring a template, mirror the LIVE paths and re-derive whether each
  piece is reachable in the new context - do not copy an exception/branch just
  because the template has it. A defined-but-never-raised exception is a small
  smell that a route/test claims coverage it does not have (the "409 dup" that
  could never fire).

## Action items

- [x] MINOR + NITs addressed before landing (dead code removed, project_id
  immutability pinned, 409/201 wording corrected).
- No new ledger entry: the "don't copy a template's dead code" observation is a
  one-off nit, kept here rather than bloating the terse ledger.
