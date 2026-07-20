# Retro: Settings backend - config override store + gated writable endpoint

- TASK: 20260720-184136
- BRANCH: feature/settings-config-store
- REVIEW ROUNDS: 1 (APPROVE, 1 MINOR routed onward + 3 NITs fixed in-round)

## What went well

- Grounding the plan in the code before following it literally paid off: the
  plan proposed a `get_settings()` provider re-read per request; counting the
  actual closures (37 `agent`/`settings` reads in create_app) and seeing the
  agent reads settings per turn made in-place mutation the clearly
  correct-and-smaller choice. `validate_assignment=True` gave free per-write
  validation, so the "smaller" option was not "weaker".
- The AgentHandle trick (a protocol-implementing wrapper) meant live
  backend/enabled switching cost ZERO churn at the 37 call sites - they all
  still hold one object that IS an Agent.
- The out-of-context review found only NITs; all three were cheap and genuinely
  improved the code (atomic persist, a drift-guard test, an aclose comment), so
  they were folded into the same round rather than deferred.

## What went wrong

- Lost ~2 tool-cycles to a worktree import-shadowing trap: bare `pytest` in the
  sprout worktree imported scufris from the MAIN checkout (the editable
  install's absolute path on sys.path), so the new `AgentHandle` symbol
  ImportError'd at collection even though mypy was green against the worktree.
  Root cause: the console-script `pytest` does not put CWD first on sys.path;
  `python -m pytest` does. Diagnosed with `inspect.getfile(scufris.agent)`.
- The whitelist ended up duplicated (store `WRITABLE_KEYS` + API
  `AgentConfigUpdate` fields). Caught by review; pinned with a sync test rather
  than over-engineered into one derived source.

## What to improve next time

- In a sprout worktree, ALWAYS run `python -m pytest` (not bare `pytest`), or
  the tests silently run against the main checkout and miss new branch symbols.
- When a plan step encodes an approach ("route reads through a provider"),
  confirm the mechanism against the code (how many readers, do they share the
  object) before implementing it - the cheaper in-place path was invisible
  until the closures were actually counted.

## Action items

- [x] Two lessons added to LESSONS.md (worktree `python -m pytest`;
      in-place-mutation-beats-a-provider-rewire).
- [x] Stale frontend "read-only" copy routed to task 5 (20260720-184148 Notes).
- No follow-up code task; T2-T6 already planned build on this.
