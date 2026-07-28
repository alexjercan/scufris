# Retro: Telegram read-only /settings subcommands + /stats

- TASK: 20260728-222321
- BRANCH: feature/telegram-settings-cmds
- REVIEW ROUNDS: 2 (round 1 out-of-context APPROVE with one MINOR; round 2
  in-session confirming the MINOR fix)

Process observations only; what/why/design live in TASK.md, the findings in
REVIEW.md.

## What went well

- Two parallel exploration agents mapped BOTH surfaces (the telegram transport
  and the web dashboard's read-only endpoints) before planning, so the plan
  could name the exact in-process readers (`agent_health`, `read_usage`,
  `_tools_for_servers`, `collector.sample`) and `/work` had essentially no
  discovery churn.
- Settling the command surface (subcommands vs a single `/settings` umbrella)
  and the `/stats` verbosity at the plan gate via `AskUserQuestion` meant I
  built the shape the user wanted the first time.
- Breaking the would-be `telegram -> app` import cycle proactively (moving the
  DTOs to a new `mcp_models.py` and re-exporting) instead of a `TYPE_CHECKING`
  hack kept the layering honest; a one-line import check confirmed it instantly.
- The out-of-context reviewer independently re-derived the exact risk the task
  had flagged (a sync provider stalling the poll loop), which is the out-of-
  context default earning its keep.

## What went wrong

- The bulk `replace_all` on the "identical" `TelegramBot(` constructor block
  missed the one site whose 5th argument was `idle_cancel` (not `on_cancel`), so
  the first test run failed on a missed constructor. Root cause: changed a
  widely-built signature without first enumerating every call site AND its
  per-site argument variations - the work skill already warns that a new
  required param breaks exhaustive constructors.
- Two test-data artifacts caused false failures (a `backend_version` that
  redundantly began with "codex"; asserting no backtick in a body whose code
  fence legitimately uses them). Root cause: wrote the substring assertions from
  my mental image of the output rather than from the actual rendered string.
- R1.1 (sync `read_usage`/`collector.sample` awaited inline on the loop) shipped
  in round 1 even though the task text itself listed "can a slow provider block
  the poll loop" as a concern I had reasoned about. Root cause: treated a
  self-identified risk as a note to carry, not a work item to close.

## What to improve next time

- Before adding/removing a parameter on a constructor built in many places, grep
  every call site FIRST and scan for per-site arg variations, so the update is
  one pass, not a re-run after a missed site.
- Print/inspect one actual rendered output before writing substring assertions
  for a brand-new formatter.
- When the TASK itself names a risk, resolve it during `/work`; a self-flagged
  concern is a work item, not a footnote for review to find.

## Action items

- [x] Ledger: `sync-read-inline-on-a-latency-loop-stalls-it`.
- [x] Ledger: `grep-every-call-site-before-changing-a-built-signature`.
- [x] Ledger: `assert-a-new-formatter-against-its-real-output`.
- No follow-up code task: R1.2 (NIT) is won't-fix by design; the deferred
  write/remote-exec work is already recorded in TASK.md Notes for a future flow.
