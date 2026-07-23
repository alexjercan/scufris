# Review: harden pending-agents poll onto a collision-proof path

- TASK: 20260723-120507
- BRANCH: fix/pending-agents-path

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (round-1 findings), in-session pass verified and recorded

The out-of-context reviewer ran the full suite (367 passed, ruff + mypy clean),
grepped the tree for stray old-path references (all remaining `/api/agents/pending`
hits are in immutable historical records - LESSONS.md, prior task files, and this
task's own explanatory prose - not executable code), and independently BOOTED the
worktree source (`python -m scufris`, CWD-first) confirming
`GET /api/pending-agents -> 200` and the vacated `GET /api/agents/pending -> 404`.
In session I had already boot-verified the same. No findings.

No BLOCKER/MAJOR/MINOR/NIT.

No open `manual:` DoD items (all proofs are `test:`/`cmd:`).
</content>
