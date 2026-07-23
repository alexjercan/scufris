# Retro: harden the orchestrator<->sub-agent session view on the agent page

- TASK: 20260723-001301
- DATE: 20260724
- OUTCOME: shipped; per-agent chat reattaches to an in-flight run on load and
  streams it to completion. `cd web && npm run ci` green (172 tests, +6). Two
  review rounds, both APPROVE.

## What this was

A bug/harden task: the per-agent detail page (`/agents/<id>`) only loaded a
settled transcript once and only rendered turns the browser POSTed itself, so an
orchestrator-driven sub-agent turn (`message_agent`/`run_agent`, run on the shared
supervisor + event bus) never showed live, and reloading/reselecting mid-turn
froze on the settled transcript. The fix reattaches to the existing backend SSE
relay `GET /api/agents/<id>/events` from the frontend; no backend change.

## What went well

- **Planning found the real root cause before any code.** An Explore pass mapped
  both chats and the backend, revealing the bug is a REGRESSION: F0
  (20260721-112428) built the `EventSource` reattach in the OLD inline run panel,
  but the F1/F2/F3 detail-page reshape dropped it while the backend `/events` relay
  survived and stayed tested. So the task was re-wiring, not new machinery - which
  the plan/DECISION captured, keeping scope tight (one cohesive frontend task).
- **Injected capability kept the pure component testable.** Modelling `reattach`
  as an injected function (like `streamTurn`/`loadTranscript`) let jsdom drive it
  without a real `EventSource`, and a `FakeEventSource` stub covered the real
  wiring end to end (status gate + stream-to-settle + close-on-terminal).
- **Reproduce-first paid off.** The red component test drove the capability shape
  before implementation.

## What went wrong / difficulties

- **The out-of-context review APPROVEd a real race I only caught by tracing the
  backend.** Round 1's design "reconcile by reloading the transcript on settle"
  looked clean, but the backend persists the (possibly new) session id in a
  post-turn `on_complete` callback that runs in the supervisor's `finally` AFTER
  the `done` frame is dispatched. So the reload could `GET /transcript` before the
  session id was registered and, for a first-ever turn, read EMPTY and drop the
  turn. The reviewer accepted the reload; I found it by reading
  `_launch_agent_turn.persist` + `supervisor._execute` ordering. Fix: settle by
  pushing the bus's `done` reply (carries text+tools+usage) exactly like a local
  turn - no reload, no race, and simpler (reattach/local share one settle path).
- **A stubborn typecheck/lint fight with the test double.** `let x: FakeEventSource
  | null` declared BEFORE the class resolved its annotation to `null` under the
  webpack ts-loader build (a forward type reference), so a `if (!x) throw` guard
  narrowed to `never`; and calling a block-scoped class method tripped
  `no-unsafe-call`. Burned several iterations. Settled on an explicitly-typed
  module-level `openedSources: FakeEventSource[]` array + a free `emitFrame` helper.
- **Streaming throttle bit a test.** The render throttle debounces the 2nd+ text
  delta ~50ms, so a two-delta assertion only saw the first within a
  `setTimeout(0)` flush - delivered the token as a single delta instead.

## Lessons (folded into LESSONS.md)

- `out-of-context-review-misses-cross-layer-timing` - an APPROVE from an
  out-of-context reviewer does not clear a race that spans the frontend/backend
  seam; when a UI reconcile depends on WHEN the backend persists, trace the actual
  callback ordering (here: a post-turn `finally` callback runs after the terminal
  SSE frame), do not trust a green suite or a reviewer who only read the frontend.
- `forward-typed-null-tracker-resolves-to-never` - a `let x: T | null = null`
  declared BEFORE class `T` resolves its annotation to `null` under the ts-loader
  build (forward type ref), so a null-guard narrows to `never`; and calling a
  block-scoped class method trips typed-lint `no-unsafe-call`. For a construction
  tracker in a test double, use an explicitly-typed module-level array declared
  and a free helper, not a `let` before the class.
- `ui-reshape-silently-drops-a-wired-capability` - when a component is replaced by
  a reshaped one, a capability wired into the OLD surface (here an SSE reattach
  EventSource) can vanish while its backend half survives and stays green. After a
  UI reshape, check each capability the old surface had is re-wired, not just that
  tests pass.

## Follow-ups

- Manual acceptance (batched to the goal's Finish): with a real backend, drive a
  sub-agent turn from the orchestrator (or a long `/chat` turn), reload/reselect
  mid-turn, confirm the transcript rebuilds and the turn keeps streaming with no
  duplicated/phantom bubbles.
- Out of scope, noted: reattach fires only at mount, so a turn that STARTS while
  the page is already open-and-idle is not shown live until a reload (the DoD
  targets reload/reselect). A status-poll-triggered reattach would close that gap.
- Out of scope, noted (R2.1): if sidebar refresh ever becomes per-agent and wires
  `onAfterTurn` on this page, note that it now fires on reattach settle too.
