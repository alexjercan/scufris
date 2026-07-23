# Review round 1: APPROVE

- VERDICT: APPROVE

The change successfully reattaches the per-agent chat to in-flight runs via the existing SSE relay, properly gated on active status and reconciling from the authoritative transcript. The runTurn refactor correctly defers bubble attachment in reattach mode to avoid phantom renders on idle runs, and the stop/streaming state machine properly guards against race conditions with locally-initiated turns. Test coverage is comprehensive for the happy paths. One defensive path (onerror handler) is unreachable in tests due to a missing static constant on the fake, but the code is production-correct and doesn't impact passing tests.

## Findings
- [x] R1.1 (NIT) web/src/agent-chat-view.test.ts:62-76 - FakeEventSource lacks a static CLOSED constant, so the onerror path at chat-stream.ts:109 never triggers in tests (comparison `readyState === EventSource.CLOSED` becomes `number === undefined`). This is harmless for the current test suite, but production reconnect-on-transient-drop is untested. Suggested fix: add `static CLOSED = 2` to FakeEventSource and add a test case that fires onerror() with readyState === CLOSED to verify finish() is called and the promise resolves. (Minor: code is production-correct for browser EventSource, just untested in jsdom.)
      RESOLVED: added `static readonly CLOSED = 2` to FakeEventSource and a test "gives up (resolves, frees the composer) when the event stream errors closed" that sets readyState=CLOSED and fires onerror, asserting the reattach resolves with no hung bubble and the composer freed.

## Findings (round-1 addendum, self-found during author verification)
- [x] R1.2 (MAJOR) web/src/agent-chat-view.ts settle reattach mode [backend: scufris/app.py _launch_agent_turn.persist + scufris/supervisor.py _execute] - The reviewer accepted "reconcile by reloading the transcript on settle", but that reload RACES the backend. The (possibly new) session id is persisted in the run's `on_complete` callback, which the supervisor runs in its `finally` AFTER the `done` frame has already been dispatched to SSE subscribers. So reattach's reload-on-settle can `GET /transcript` before the session id is registered; for a FIRST-ever turn (no prior session id) that read returns an EMPTY transcript, dropping the very turn just streamed - violating the DoD ("shows the full transcript AND continues streaming"). Traced concretely, not hypothesized.
      RESOLVED: reattach now SETTLES like a local turn - push the `done` frame's reply (which carries text + tool_calls + usage) into the log, no transcript re-fetch. The turn's prompt line comes from the mount-time transcript load (the backend writes the user message at turn start). Removed the now-unused `reloadTranscript`. DECISION.md point 3 rewritten with the rationale; tests updated to assert push-on-settle (loadTranscript called once, one new assistant bubble, no reconcile fetch).

## Verdict after fixes: APPROVE

Both findings resolved on the branch. `cd web && npm run ci` green (172 tests). The
reattach path is simpler after R1.2 (reattach and local turns share one settle
path; the only reattach-specific behavior is the lazy bubble).


# Review round 2: APPROVE

The revision correctly unifies settle behavior for local and reattached turns, eliminating the race that R1.2 identified. The refactored `runTurn` defers bubble attachment in reattach mode until a frame arrives (`ensureBubble()`), ensuring idle runs do not disable the composer or render a phantom bubble. The `.then()` no-terminal path properly handles all three edge cases: settled turns, idle reattaches, and defensive dangling bubbles. Test coverage now comprehensively proves push-on-settle and exercises the onerror-on-CLOSED path.

## Findings

- [x] R2.1 (MINOR) web/src/agent-chat-view.ts:543 - `config.onAfterTurn?.()` is now called on reattach settle, while it was not before. For the orchestrator landing page (agent-view.ts:205), `onAfterTurn` refreshes the sidebar. For the per-agent detail page (startAgentChat line 869+), `onAfterTurn` is not set, so this is a no-op. Semantically safe: a reattached orchestrator-driven turn is neither a fresh local turn nor a per-agent turn, so calling the orchestrator's refresh would be incorrect. However, this is an implicit contract: if a future caller sets `onAfterTurn` on the per-agent page and relies on it NOT being called during reattach, the behavior will silently change. Mitigation: none needed for current code, but worth documenting in a follow-up if sidebar refresh becomes per-agent (out of scope here).

## Verdict: APPROVE

All unified settle paths are sound. Bubble attachment is correctly deferred in reattach mode, preventing double-render and composer lockup. The mount-time transcript load provides the in-flight turn's user prompt; no race from skipping the reconcile reload. Tests prove push-on-settle and comprehensive edge cases. `npm run ci` passed (172 tests green).
RESOLVED: added a clarifying comment at settle noting onAfterTurn fires on every settle (reattach included) but is a no-op on the per-agent page, which does not set it.
