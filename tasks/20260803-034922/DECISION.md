# Decision: Delete the negative usage case rather than restate it (R2.2)

- DATE: 20260803-170314
- STATUS: ACCEPTED
- TASK: 20260803-034922
- TAGS: tests, agents, frontend

## Context

Round 2 of 20260801-100415 flagged `web/src/agent-view.test.ts:386` ("hides the
meter when the backend cannot report usage") as a test that passes with or
without the `quota.value` unwrap in `loadUsage` (`web/src/agent-view.ts:145`).
The finding offered two remedies: assert something the envelope discriminates
(the meter is empty as well as hidden), or drop the case as covered by the
supported one. `renderUsage` (`web/src/chat-sidebar.ts:165`) reads
`usage?.primary` and, when it is absent, calls `meter.replaceChildren()` AND
sets `hidden = true` - so for `{supported: false, value: null}` the unwrapped
value and the raw envelope produce a byte-identical DOM. Neither the hidden
flag nor emptiness can separate them.

## Decision

Delete the "hides the meter when the backend cannot report usage" case. The
unwrap is pinned by the case directly above it, "renders the meter from a
supported envelope's value" (`agent-view.test.ts:375`), which is the only shape
where the two readings differ. Verified in scratch: replacing
`renderUsage(quota.value)` with `renderUsage(quota as unknown as UsageQuota)`
turns exactly that case red (`meter.hidden` true, expected false) while the
negative case stays green - 1 failed, 6 passed. Extend the surviving case's
comment to record that it is the unwrap's pin, so the next reader does not
re-add an unfalsifiable twin.

## Alternatives considered

- **Assert the meter is empty as well as hidden** (the finding's first option).
  Does not discriminate: `replaceChildren()` runs on both the null value and
  the primary-less envelope, so the assertion is green either way. Rejected as
  the same non-falsifiable test with more assertions.
- **Restate as `{supported: true, value: {primary: null, secondary: {...}}}`**
  (NOTES.md's fallback). Also does not discriminate: unwrapped gives
  `primary === null` and raw gives `primary === undefined`; both hide the meter.
  Rejected for the same reason.
- **Keep the case as-is** and mark the finding won't-fix. Costs nothing today
  but leaves a test whose name promises a guarantee it does not hold, which is
  the exact defect this task exists to remove.

## Consequences

- The frontend loses explicit coverage of `supported: false` reaching
  `renderUsage`. That path is a subset of `usage == null`, which the surviving
  case's negative branch exercises through `renderUsage`'s own unit coverage;
  no product behaviour goes unpinned.
- Test count drops by one, so `web/src/agent-view.test.ts` reads as slightly
  thinner. The tradeoff is honest: a deleted vacuous test is worth more than a
  green one that cannot fail.
- If `renderUsage` ever grows a distinct rendering for "backend has no usage
  reader" versus "no quota yet", that new behaviour needs its own case - this
  decision does not forbid one, it removes the one that tested nothing.
