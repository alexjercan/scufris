# Retro: Agent page - tools/model panel + tool-call & token display

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- The backend task did the hard part (parsing tool-calls + usage), so this was a
  clean consume: mirror the types in `common.ts`, fetch the two endpoints, and
  render. The spike's decisions (per-turn summary, input_tokens as context) held.
- Extracting pure render helpers (`renderAgentPanel`, `messageMeta`,
  `applyUsage`) let the new UI be jsdom-tested without mocking fetch - the same
  side-effect-free pattern used for stats/processes. The `_resetAgentState` hook
  (from the process-view lesson) kept the cumulative-token module state from
  leaking across test cases.
- Additive UI: the chat still renders text-first, so a reply with no tools/usage
  simply shows no meta line - nothing regressed for the plain-chat path.

## What went wrong / friction

- Nothing notable. The one judgement call (what the cumulative indicator means)
  followed the spike: cumulative OUTPUT tokens + last-turn `input_tokens` as the
  context signal, since the CLI exposes no per-model window.

## Lessons

- (No new lesson - this reused `side-effect-free-module-for-jsdom-tests`,
  `persistent-ui-state-needs-a-test-reset-hook`, and
  `escape-only-host-strings-in-element-content`; a sign those are now the settled
  frontend pattern.)

## Follow-ups

- Optional (deferred in the spike): a static per-model context-window map for a
  "% of context used" bar; SSE streaming for live "tool running..." feedback.
- Remaining backlog: sparkline history (tatr 20260719-182915).
