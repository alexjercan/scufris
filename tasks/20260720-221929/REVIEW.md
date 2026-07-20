# Review: A1 AgentStore (agent as a first-class record + CRUD)

- TASK: 20260720-221929
- BRANCH: feature/agent-store

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Check suite (reviewer ran it in the worktree): ruff clean, mypy clean (33
files), `python -m pytest` = 229 passed. Verified independently in-session.
Assessed as a faithful mechanical mirror of `projects.py`: atomic persist,
tolerant load, `settings_writable` gate first in every mutator, FK validation
(`create` -> `ProjectStore.get` -> `InvalidAgent` -> 422), `update` omits
project_id/session_id/state, `AgentUpdate` `extra="forbid"`, route gate ordering
(403 before 404/422), deferred `/run` + `/events` correctly absent, each new
route has its own 403/404/422 test.

- [x] R1.1 (MINOR) scufris/agent_store.py - `DuplicateAgent` is defined-but-dead:
  `_unique_id` guarantees `create` never collides, so it is never raised, never
  imported, and no 409 branch/test exists. The TASK Steps 1/3 mention it and
  "409 dup". Drop the dead class and the 409 language (cleaner than keeping
  vestigial parity with projects' also-unreachable `DuplicateProject`).
  - Response: Fixed. Removed the `DuplicateAgent` class; corrected TASK.md Steps
    to drop the 409/DuplicateAgent language (create only 200/422, gate 403).
- [x] R1.2 (NIT) tests - no test pins the "update cannot change project_id"
  guarantee; `AgentUpdate`'s `extra="forbid"` makes `{"project_id": ...}` a 422,
  worth a one-line regression pin.
  - Response: Fixed. Added an assertion to `test_agent_create_validation` that
    `PATCH /api/agents/{id}` with `project_id` returns 422.
- [x] R1.3 (NIT) tests/spec - the "201" wording in the plan is aspirational;
  FastAPI POST returns 200 here (consistent with `create_project`).
  - Response: Corrected the TASK.md wording to 200 (matches `create_project`; no
    code change - 200 is the intended, consistent behavior).

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (round 1 was already APPROVE; these are dead-code removal
  + a one-line regression test + a doc-wording fix, no behavioral change)

Verification: `DuplicateAgent` removed (grep confirms no remaining reference in
scufris/*.py or tests/*.py); the new immutability assertion (added to
`test_agent_create_validation`, so the count stays 229) passes and would fail if
`AgentUpdate` dropped `extra="forbid"`. Suite: ruff + mypy clean, 229 passed. No
new findings.
