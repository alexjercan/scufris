# Review - Dedupe tool-call chips in assistant meta line

## Round 1 - out-of-context reviewer

Verdict: **APPROVE**

Scope reviewed: the staged diff on `fix/dedupe-tool-chips`
(`web/src/agent-chat-view.ts`, `web/src/agent-chat-view.test.ts`, `TASK.md`).

Confirmations:

- `[...new Set(names)]` preserves first-occurrence order (ECMAScript guarantees
  Set iteration = insertion order; a duplicate insert does not reorder). So
  `["a","b","a","c"] -> ["a","b","c"]`.
- Both display paths deduped and now agree: settled chips (`messageMeta`) and
  the live `ran ...` status (`paintStatus`).
- No other meta behavior changed (ran label, token count, empty-line guard
  untouched).
- Tests: the polling-turn case asserts exact chip text and order for a realistic
  repeated sequence; the `distinctTools` unit test covers interleaved repeats.

Minor, non-blocking (no change requested):

- `paintStatus` re-derives the distinct list each 500ms tick / `onTool`; O(n)
  over a tiny array, irrelevant.
- No direct DOM test of the live `paintStatus` dedupe; the logic is shared via
  the unit-tested `distinctTools`, so risk is low. Left out to keep scope tight.

No correctness bugs, regressions, or missed edge cases within scope.
