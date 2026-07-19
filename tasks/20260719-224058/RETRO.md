# Retro: agent tatr tools

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- Small, self-contained backend win built on the settled MCP patterns (fixed-arg
  `_run`, return-text-not-raise, integration tests against real `tatr` in a temp
  dir). Live-verified end to end.
- Documenting the filter language INSIDE the `tatr_ls` tool description is the
  right move - the model reads tool descriptions, so that is where the knowledge
  has to live for `-f` to actually get used. Cheap, high leverage.
- Crossing the read-only line deliberately: `tatr_new` is the first write tool, so
  the server docstring was updated to say so and why it is safe, rather than
  leaving a stale "read-only" claim.

## What went wrong / friction

- The two-task sort/filter test would have flaked on tatr's second-resolution IDs
  (a same-second `new` fails). Caught it from the tatr SKILL.md note before it bit,
  and spaced the creates with `sleep(1.1)`. Worth a ledger line so future tatr
  tests do not learn it the hard way.

## Lessons

- `tatr-ids-are-second-resolution` (frontend/testing): tatr task IDs are
  `YYYYMMDD-HHMMSS`, so two `tatr new` in the same second COLLIDE (the second
  fails "already exists"). Any test or tool that creates multiple tasks must space
  them (`sleep(1.1)`) or expect+retry the collision. 20260719-224058.

## Follow-ups

- A `tatr_edit` companion (move status/priority) is an easy later add if the agent
  needs to advance tasks, not just create them.
- Next in this batch: delete a conversation (20260719-224100), then fork
  (20260719-224101).
