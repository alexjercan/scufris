# Retro: Orchestrator logs food from plain language (steer STEERING_PREAMBLE)

- TASK: 20260727-020723
- BRANCH: feature/orchestrator-food-logging-steer
- REVIEW ROUNDS: 1 (APPROVE, out-of-context, no findings)

## What went well

- Understanding-first paid off immediately: a `grep` of LESSONS.md for
  "steering/preamble" surfaced three directly-applicable lessons
  (`codex-tool-choice-only-steers-via-the-turn-prompt`,
  `orchestrator-steering-is-one-block-two-clauses`,
  `ground-steering-text-in-the-real-tool-signatures`) that pinned the fix
  shape exactly and pre-empted the two classic mistakes (a second sentinel
  block that `strip_steering` would leak, and a typo'd tool name that steers
  to a dead call). The diagnosis - "the tools exist, only the steering is
  missing" - fell straight out of the existing host-tools clause pattern.
- Test-first worked as intended: wrote the two assertions, saw the food-chain
  test go red for the right reason (`macros_lookup not in steered`) BEFORE
  adding `_JOURNAL_CLAUSE`, then green after. The revert-fails-the-test
  property is therefore real, not assumed.
- The out-of-context reviewer independently re-derived the load-bearing claims
  (revert breaks the test; all 11 tool names exist verbatim in mcp_server.py;
  the `macros_lookup` row is byte-for-byte what `journal_add_macros` accepts)
  and returned a clean APPROVE in one round.

## What went wrong

- `tatr new -b <body-file>` produced DUPLICATED header lines: the body file I
  passed started with its own `- STATUS: / - PRIORITY: / - TAGS:` block, and
  tatr ALSO injects those from the title/flags, so the created TASK.md had the
  three lines twice and needed a hand edit. Root cause: assumed the body file
  is the whole file; in fact tatr owns the header and the body starts below it.

## What to improve next time

- When creating a task with `tatr new -b <body-file>`, the body file must NOT
  repeat the `STATUS/PRIORITY/TAGS` header - start it at the first `##`
  section (or the Goal). Let tatr write the header from `-p` / `-t` / the
  title. (New ledger entry: `tatr-new-body-file-omits-the-header`.)

## Action items

- [x] Ledger: added `tatr-new-body-file-omits-the-header` (x1).
- (no follow-up code tasks; the change is self-contained)
- Pending manual acceptance (operator): DoD #4 live "log that I had 2 eggs"
  turn against the real the-den + macros DB - batched at flow Finish.
