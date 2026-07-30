# Review: Add the dashboard host approval queue and audit surface

- TASK: 20260730-104520
- BRANCH: feat/host-approval-queue

## Round 1

- VERDICT: REQUEST_CHANGES
- REVIEWER: in-session (no out-of-context mechanism available - subagents are
  disabled in this session, so the round-1 default could not be used. Recorded as
  the exception the review skill allows. Both MAJOR findings below were produced by
  PROBING the built view in jsdom rather than by reading it, and the probe output is
  quoted verbatim.)

What was verified rather than taken on trust:

- `cd web && npm run ci`: prettier, eslint, 247 vitest tests, webpack build - all
  pass. `python -m pytest`: 841 passed. `nix flake check`: all four checks pass.
  `nix build .#scufris .#web`: exit 0, and `host/index.html` + `host.js` are in the
  built `.#web` output, so the page ships rather than only building.
- The page was SERVED for real (`python -m scufris serve` with the built bundle):
  `/host/` 200, `/host.js` 200, and the two endpoints it reads were curled. That is
  what caught the not-configured signal bug recorded in NOTES.md - the whole test
  suite was green with it in.
- The escaping test was checked for the ability to FAIL: switching `text()` to
  `innerHTML` turns it red (one test), so it is not a test that cannot fail.
- The auth test was run against a real static mount: `/host/` redirects to the
  login page unauthenticated, `/host.js` is 401, and it serves 200 with a session.

- [x] R1.1 (MAJOR) web/src/host-view.ts:722 (`refresh`) with
  web/src/host-view.ts:560 (`renderHost` calling `root.replaceChildren()`) - the
  4-second poll destroys what the operator is typing, which makes the one-way
  approve path effectively unusable. `renderHost` rebuilds the whole page, so every
  poll replaces the acknowledgement input and the deny-reason input with fresh empty
  ones. Probed in jsdom: type `gc_sto` into `.host__ack`, focus it, then render
  again as a poll would -> `PROBE typed value after a poll: ""` and
  `PROBE focus kept: false`. Typing `gc_store` (8 characters) inside a 4-second
  window that also steals focus mid-word is not a thing to ask of an operator on a
  phone. Suggested change: skip a POLL-driven re-render while the operator is
  interacting - export a small `isTyping(root)` (an input/textarea inside `root`
  holding focus) and have `refresh` update `data` but defer the render until the
  next poll; a decision-triggered reload still renders, because the controls are
  gone by then. Pin `isTyping` directly, and pin that a re-render after a decision
  still happens.
  - Response: fixed. `isTyping(root)` reports whether an input or textarea inside the
    page holds focus, and the POLL passes `{poll: true}` so its render is held back
    while that is true - the data still refreshes on every tick, so nothing goes
    stale behind the deferral, and a decision-triggered render ignores the flag
    because its controls are gone by then. Pinned by
    "never re-renders over what the operator is typing", which also asserts that a
    focused BUTTON does not count (a poll should still refresh then).
- [x] R1.2 (MAJOR) web/src/host-view.ts:118 (`lastError`) - the error banner never
  clears, so after any failed decision the page keeps telling the operator about it
  forever. `dispatch` sets `lastError` on failure and nothing ever resets it, while
  `renderHost` renders `data.error ?? hostError()` on every pass. Probed: after a
  refused approve the banner reads `409 already decided`, and after a LATER
  successful deny it still reads `409 already decided`. On a page whose whole job is
  to say truthfully what happened to this machine, a permanently displayed stale
  failure - next to a queue that has since moved on - misinforms about who decided
  what. Suggested change: clear `lastError` at the start of a successful `dispatch`
  (and on a successful refresh), and pin it with a test that a success after a
  failure leaves no banner.
  - Response: fixed - `dispatch` clears `lastError` on success, pinned by
    "clears the error banner once something succeeds", which reproduces the probe:
    refused approve -> banner, successful deny -> no banner.
- [x] R1.3 (MINOR) web/src/host-view.ts:672 (`readAudit`) - a non-503 failure of the
  audit read is swallowed (`return { configured: true, rows: [], detail: "" }`), so
  a 500 renders "the helper's log is empty: nothing has been asked of it yet" - a
  blank that reads as fine, which is the exact failure mode this repo's host package
  was built to avoid. Suggested change: distinguish "read failed" from "empty" and
  say which, the way the preview's availability line does.
  - Response: fixed - `AuditRead.failed` carries the reason, the table renders it in
    the `host__unavailable` style instead of the empty-log sentence, and a test
    asserts the two are not confused.
- [x] R1.4 (MINOR) web/src/host-view.ts:153 (`formatExpiry`) - the signature mixes
  units: `expiresAt` is unix SECONDS and `now` is MILLISECONDS, and `staleReason`
  ten lines away uses the other convention (`expires_at * 1000 <= now`). It is
  correct today and tested, but it is one careless call site away from a wrong
  countdown on the field that tells the operator how long they have. Suggested
  change: take both in milliseconds and convert once at the boundary.
  - Response: done - `formatExpiry(expiresAtMillis, nowMillis)` takes one unit, and
    `expiryMillis(record)` is the single place the wire's unix-seconds field is
    converted; `staleReason` uses it too, so the two conventions are now one.
- [x] R1.5 (NIT) web/src/style.css `.host__audit th, .host__audit td` - `white-space:
  nowrap` on every cell means one long argv (a store path) makes the whole table
  scroll sideways even at desk width. Suggested change: keep nowrap on the narrow
  columns (when, event, outcome) and let `what` wrap.
  - Response: done - only the command column wraps (`break-all`, with a min-width so
    it does not collapse).

Pending user checks (not resolved by this review):

- manual: the queue is readable at phone width and the risk difference between a
  service restart and a system switch is obvious at a glance. NOTE: this session
  has no browser tooling, so the visual render is unverified by me - the structure,
  the classes and the media query are in place and tested, but someone has to look
  at it.

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (same exception as round 1. Each fix was re-verified by
  running its pin, and the two MAJORs were re-derived by replaying their probes as
  assertions rather than by reading the patch.)

All five findings are resolved and ticked.

- R1.1 re-verified: the probe that showed a typed token and the focus vanishing is
  now a test of `isTyping`, and the poll consults it. The fix would fail the pin if
  reverted, because the unguarded render is what emptied the input.
- R1.2 re-verified by the same sequence the probe ran: refused approve leaves the
  banner, a later successful deny clears it.
- R1.3, R1.4 and R1.5 verified by their own tests and by reading the rendered
  table.

Gate after the fixes: `cd web && npm run ci` - prettier, eslint, 251 vitest tests
and the webpack build all pass; `python -m pytest` 841 passed; `nix flake check` all
four checks; `nix build .#scufris .#web` exit 0 with the page in the output.

Pending user checks (not resolved by this review):

- manual: the queue is readable at phone width and the risk difference between a
  service restart and a system switch is obvious at a glance. This session has no
  browser tooling, so the visual render is unverified by me: the structure, the
  classes and the media query are tested, but someone has to look at it.
