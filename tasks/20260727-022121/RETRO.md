# Retro: Auto-delegate task implementation to a backend sub-agent (steering)

- TASK: 20260727-022121
- BRANCH: feature/delegate-agent-steer
- REVIEW ROUNDS: 1 (APPROVE, out-of-context, 2 no-change NITs)

## What went well

- Second run of the steering-clause pattern in a row (after 20260727-020723),
  and it is now effectively a playbook: grep LESSONS.md for the steering trio,
  add a clause to the ONE sentinel block, ground every tool name verbatim
  against mcp_server.py, and prove it with presence + single-block + strip
  round-trip tests. The cycle was fast because the shape was known.
- Understanding-first surfaced the real load-bearing fork instead of a guess:
  the flow skill is a Claude Code skill, so a codex sub-agent literally cannot
  load it - "steer the sub-agent to use flow" and "one path for both backends"
  are mutually exclusive. That went to the user (AskUserQuestion) and into a
  DECISION.md, per the flow "name the constraint that makes options exclusive"
  rule - not silently resolved.
- Connected the steering fix to the ACTUAL reported bug by reading the code:
  create_agent defaults to permission_mode="manual", and enums/backends confirm
  manual = read-only, so a delegated agent with the default can make 0 tool
  calls no matter how good its goal. That turned a vague "make delegation work"
  into a concrete steer (demand edit/auto) with a verifiable premise.
- Last cycle's lesson applied forward: the `tatr new -b` body file omitted the
  STATUS/PRIORITY/TAGS header, so no duplicated-header hand-fix this time.

## What went wrong

- Nothing material. Minor friction: `tatr edit --status` does not touch the
  Flow State marker, so FLOW STEP had to be hand-edited separately at each
  transition (WORKING after sprout, COMPOUNDING after APPROVE). This is the
  known two-surface split (tatr owns STATUS; the TASK.md body owns FLOW STEP),
  not a mistake, but it is easy to forget the second edit.

## What to improve next time

- Keep treating the steering-clause change as a checklist (clause in the one
  block, verbatim tool names, three assertions). It converged in one review
  round twice now.
- When a delegated/spawned-agent feature is in play, always check the
  permission-mode default first - a read-only default is an invisible
  "did nothing" cause that no amount of prompt wording fixes.

## Action items

- [x] Ledger: bumped the steering trio occurrence counts with this task id and
      added `steer-permission-mode-for-implementing-agents` (x1).
- (no follow-up code tasks; steering-only scope, self-contained)
- Pending manual acceptance (operator): DoD #5 live "implement task X using
  codex" delegated run - batched at flow Finish.
