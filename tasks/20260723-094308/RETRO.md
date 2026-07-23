# Retro: BC3 pending_agents + acknowledge orchestrator MCP tools

- TASK: 20260723-094308
- BRANCH: feat/pending-agents
- REVIEW ROUNDS: 1 (APPROVE, clean)

## What went well

- This was the easy dividend BC2's `DECISION.md` (Option B, role model) promised:
  because tools are scoped by AUDIENCE, the two new orchestrator tools needed ZERO
  scoping work - not being in `_AGENT_ROLE_TOOLS` makes them orchestrator-only for
  free, and the existing `test_apply_role_agent_keeps_only_request_input` (exact
  `== {"request_input"}`) already proves a sub-agent can't reach them.
- Store-first again: the only real design choices were the pending-set predicate
  and the route ordering; building `pending_outcomes`/`acknowledge` first made the
  tool + endpoint layers thin, obvious glue.
- Followed the existing `/api/agents/backends` precedent for the static-before-
  param route ordering instead of rediscovering the trap - and pinned it with a
  round-trip test that a shadowed route would fail.
- Clean out-of-context APPROVE: the reviewer independently renamed the route and
  mutated the predicate to confirm both tests genuinely fail.

## What went wrong

- Minor self-inflicted: a non-ASCII word slipped into TASK.md prose (a stray CJK
  "directly"), caught by the pre-commit non-ASCII grep before it landed. The
  existing `grep-touched-files-for-non-ascii-before-commit` lesson already covers
  it - the reflex worked.

## What to improve next time

- Nothing structural. Keep running the non-ASCII grep on touched files before
  commit (it earned its keep again).

## Action items

- [x] Ledger: `static-route-before-param-route-or-it-is-shadowed` (x1) - the
  route-ordering trap, generalized so the next `/api/agents/<word>` route does not
  rediscover it.
- No follow-up code tasks. BC4 (wake bridge) and BC5 (e2e) remain, both depending
  on this + BC1/BC2.
</content>
