# Retro: agent page - context breakdown + weekly-usage panel

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- Frontend-only-ish: the backend endpoints from tatr 212203 already returned the
  data, so this was mostly two pure render helpers + wiring into the existing
  `refreshSidebar` cadence. The settled patterns (exported pure helpers, jsdom
  tests, escape-everything) applied cleanly again.
- The review earned its keep. It caught a real over-counting bug that the happy-
  path fixtures hid: the context bar used `total_token_usage.input_tokens`
  (cumulative), so a 2-turn session read ~23% when the true occupancy was ~6%.
  Checking REAL multi-turn data (`total_in=58458` vs `last_in=15263`) both
  surfaced the bug and gave the fix its regression numbers.
- Hiding panels via `element.hidden` interacted with `.usage-block { display:flex }`
  - flex beats the UA `[hidden]` rule, so the block would not actually hide.
  Caught by writing the "hides when null" test first and watching it fail;
  `.usage-block[hidden] { display:none }` restored it.

## What went wrong / friction

- The context-fill bug traces back to a loose choice in the backend task
  (212203): "context = token usage" without distinguishing cumulative-vs-current.
  On a single-turn session `total == last`, so the live smoke there looked right
  and the gap only showed on a multi-turn session. Lesson: when a number will be
  divided by a capacity (a %), verify it on data where cumulative and current
  diverge, not just a one-shot.

## Lessons

- `codex-total-vs-last-token-usage`: codex's `token_count.info` has BOTH
  `total_token_usage` (cumulative across turns, grows unbounded) and
  `last_token_usage` (the last request). For "how full is the context window"
  use `last_token_usage.input_tokens / model_context_window`; `total_*` overcounts
  and can exceed the window. Verify percent-of-capacity figures on MULTI-turn
  data where the two diverge. 20260719-212207.
- `flex-display-defeats-the-hidden-attribute` (frontend): a rule like
  `.block { display:flex }` overrides the UA `[hidden]{display:none}`, so
  `el.hidden = true` won't hide it; add `.block[hidden]{display:none}`. Pin with a
  "hides when empty" test.

## Follow-ups

- The agent-page expansion spike (212152) now has one task left:
  20260719-212208 (MCP reach - config-driven server registry + more Scufris
  tools), which is independent of the sessions/context/usage family.
- Non-blocking: usage/context refresh only on a turn (codex emits the numbers
  mid-turn); acceptable "as of last turn" semantics.
