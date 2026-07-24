# Review: Reflect the in-flight orchestrator session on the landing after refresh

- TASK: 20260724-152230
- BRANCH: fix/orchestrator-landing-reflect

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

The out-of-context reviewer diffed against master, read the shared component
(`createAgentChat` mount flow, `runTurn`/`settle`/`onUserPrompt`), compared the
landing `reattachOrchestrator` to the per-agent `startAgentChat` reattach, and
ran the full suite. Confirmed: welcome fallback on empty current; idle reattach
is a no-op (no phantom bubble); mount ordering renders the injected prompt before
the pending bubble; landing mirrors the per-agent reattach (same gate, same
no-settle-refetch); current-session and run-bus refer to the same turn mid-turn
(task 1 records the in-flight session as current); disabled-agent path is a
non-issue (`createAgentChat` returns before `loadTranscript`); both new tests are
revert-sensitive; the deferred `onSessionStarted` is honestly out of DoD scope.

Check suite (worktree): `vitest` 180/180; `npm run lint` clean; `npm run build`
OK; `nix flake check` all pass; DoD no-settle-refetch grep confirmed.

- [x] R1.1 (MINOR) web/src/agent-view.ts - `/api/agent/sessions` is fetched twice
  on mount (`loadCurrentTranscript` + `refreshSidebar`->`loadSessions`), both
  deriving `currentSessionId` from the same `data.current` (consistent value, no
  correctness bug) - a redundant round-trip.
  - Response: WONTFIX (accepted tradeoff). The two calls serve distinct seams -
    `loadTranscript` (the chat's mount hook, owned by `createAgentChat`) and
    `refreshSidebar` (the sidebar's own lifecycle). Collapsing them would couple
    the chat-mount to the sidebar render or thread a shared cache through, which is
    more coupling than one extra idempotent GET is worth. Documented here as an
    accepted cost of the clean seam separation.
- [x] R1.2 (NIT) web/src/agent-view.test.ts - dead `transcriptLoads` counter
  (incremented then discarded via `void`); `FakeEventSource` duplicated from
  agent-chat-view.test.ts.
  - Response: fixed the dead counter (removed it and the `void`; the single-fetch
    assertion already uses `calls.filter(...)`). Left `FakeEventSource` duplicated
    per test file for isolation (a shared test helper is a separate cleanup, not
    this task); NOTES discloses it.

### In-session supplement (load-bearing re-derivation)

Independently re-derived the test teeth: reverting `loadTranscript` back to
`() => Promise.resolve([])` fails the auto-open test (no "earlier q/a"); removing
`reattach: reattachOrchestrator` fails the live-reattach test (no EventSource
opened). Both fail at their own boundary - not vacuous.

Open `manual:` DoD item (pending user acceptance, batched at flow Finish):
- On the codex orchestrator landing, refresh mid-turn -> the current session
  auto-opens and the live turn keeps streaming (reply + prompt), no reload.
