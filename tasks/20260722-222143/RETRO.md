# Retro: Telegram frontend for Scufris - orchestrator-as-the-whole-UI (umbrella)

- TASK: 20260722-222143
- CHILDREN: 7 (spike + T1-T5 + CRUD extra), all CLOSED and landed
- SPAN: two /flow runs (2026-07-22 foundation, later bot + polish; closed 2026-07-26)

Per-task process retros live on each child. This is umbrella-level only.

## What went well

- Spike-first paid off: the direction was genuinely fuzzy (scope cut, control-tool
  set, orchestrator-only scoping, keep/drop on the 8 MCP tools, transport/auth),
  and deciding all five up front let T1-T5 be planned cleanly with real deps.
- Splitting the goal into a T1-T3 milestone and deferring T4-T5 was the right
  call: the orchestrator-only foundation is independently valuable and testable
  without any outward-facing bot pieces, so the risky network/token surface stayed
  gated until the foundation was solid.
- Orchestrator-only scoping held as the load-bearing invariant across every task:
  control + CRUD tools reach the orchestrator only, regular agents never receive
  them, and that is test-backed (T1's argv-level real-spawn proof especially).
- The umbrella TASK.md was kept honest across both runs - the "Run status
  (2026-07-22)" milestone note and the later "Run status (2026-07-26)" close note
  make the two-run history and the deferred/landed split legible after the fact.

## What went wrong

- The umbrella carried an OPEN done-definition (T4-T5) for four days across two
  /flow runs. That is fine by design, but the SPIKE Q4 open question (dropping
  `tatr_new` means the orchestrator needs a write-capable permission mode to
  create tatr tasks via Bash) rode along as a loose thread to reconfirm at wiring
  time rather than being resolved when it was raised.
- Follow-on polish (live turn streaming 20260726-201901, MarkdownV2 rendering
  20260726-205809) landed AFTER the original T1-T5 seed and outside this
  umbrella's done-definition. Correctly recorded as trail-only, but it means the
  "delivered" surface is larger than the umbrella's own DoD - a reader must follow
  the trail to see the real final shape of the bot.

## What to improve next time

- When a spike raises an open question that only bites a LATER task (like Q4's
  permission-mode concern), seed it as an explicit checkbox on that later task's
  Steps at seed time, not just as prose carried in the umbrella - so it cannot be
  silently skipped when the task is picked up.
- For multi-run umbrellas, keep the "what actually shipped" trail (post-DoD polish)
  in one place on the umbrella, as this one did - that pattern is worth repeating.

## Action items

- No new code follow-ups. All children CLOSED; post-DoD polish already tracked as
  its own tasks. No lessons ledger promotions surfaced at the umbrella level
  (per-task lessons already folded by each child's retro).
