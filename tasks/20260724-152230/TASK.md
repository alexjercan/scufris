# Reflect the in-flight orchestrator (codex) session on the landing after refresh (auto-open current + reattach)

- PRIORITY: 83
- TAGS: agents, sessions, frontend, codex, streaming
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As someone on the orchestrator landing (codex), when I refresh the page mid-turn
I want the current in-flight session to auto-open and its live reply to keep
streaming, so that a refresh does not drop me back to the welcome screen while a
turn is running.

Part of umbrella 20260724-151911. Scope: codex orchestrator landing.
Depends on: 20260724-152157 (the session must be recorded at turn-start first).

## Context (investigated)

The orchestrator landing (`web/src/agent-view.ts`) wires `createAgentChat` with
`loadTranscript: () => Promise.resolve([])` (line 173) and NO `reattach`
(lines 160-205) - so on mount it shows the welcome state and only loads a
session's transcript when the user clicks it in the switcher (`switchSession`,
line ~105). The per-agent page got a reattach path in Q1-A (`startAgentChat` in
`web/src/agent-chat-view.ts:904-933`, driven by `/api/agents/{id}/status` +
`/api/agents/{id}/events`), but the landing has no equivalent.

Once task 20260724-152157 records the session at turn-start, `/api/agent/sessions`
returns it as `current` on refresh, so the switcher lists/highlights it. This
task makes the MAIN chat panel reflect the in-flight turn on refresh.

## Verify-first

- [x] Confirm the orchestrator run is reachable via `/api/agents/orchestrator/status`
      and `/api/agents/orchestrator/events` (the reserved ORCHESTRATOR_ID agent -
      `agent_run_status` and `agent_events` handle it) OR determine the correct
      orchestrator run/events endpoints. Read `scufris/app.py` route table; do not
      assume. Record the endpoints in NOTES.md before wiring the frontend.

## Steps

- [x] On the landing (`web/src/agent-view.ts`), when `/api/agent/sessions` returns
      a non-null `current` on mount, auto-select it: load its transcript
      (`/api/agent/session/{id}`) into the chat instead of the empty welcome, so a
      refresh mid-turn (or after) shows the conversation - mirroring `switchSession`
      but without the POST switch.
- [x] Add a `reattach` to the landing's `createAgentChat` config that, gated on the
      orchestrator run being live (status queued/running via the confirmed endpoint),
      subscribes to the orchestrator run's event stream and drives the same handlers
      - reusing `subscribeEvents` and the Q1-A `onUserPrompt` prompt injection so the
      live reply + the driving prompt both render. No transcript re-fetch on settle
      (same session-id write-race invariant as the per-agent page).
- [x] DEFERRED (not in the DoD): wiring the landing's `onSessionStarted` (task 1's
      handler) to update `currentSessionId` live for a turn STARTED in this tab.
      `createAgentChat`'s `runTurn` does not forward `onSessionStarted` to the
      config, so this needs threading a new per-turn handler through the shared
      component; `onAfterTurn -> refreshSidebar` already re-pins the id post-turn,
      so the only gap is a fork DURING a fresh turn (a pre-existing edge case).
      Left out to keep this task focused on the DoD (auto-open + reattach on
      refresh). Filed as a follow-up note in NOTES.md.
- [x] Frontend tests (`web/src/agent-view.test.ts` or the shared harness): (a) on
      mount with a live orchestrator run + current session, the transcript loads and
      the live turn reattaches (streaming bubble appears); (b) with no active run,
      the current session's transcript loads but no phantom live bubble; (c) no
      settle-time transcript re-fetch.
- [x] Run the full gate: `nix flake check` and `cd web && npm run ci`; both green.

## Definition of Done

- On the codex orchestrator landing, refreshing mid-turn auto-opens the current
  session and the in-flight turn keeps streaming (reply + prompt bubble), with no
  wait-for-finish and no reload (test: landing mount reattaches to a live run;
  manual: real codex turn).
- Refreshing when the current session is idle opens that session's transcript
  with no phantom live bubble (test).
- No settle-time transcript re-fetch is introduced (cmd: `grep -n "reattach\|loadTranscript" web/src/agent-view.ts` shows no settle-time fetch).
- Full gate green (cmd: `nix flake check`) and web green (cmd: `cd web && npm run ci`).

## Notes

- Key files: `web/src/agent-view.ts` (landing wiring), `web/src/agent-chat-view.ts`
  (the shared component's reattach/runTurn/onUserPrompt from Q1-A),
  `web/src/chat-stream.ts` (`subscribeEvents`, `onSessionStarted` from task 1),
  `scufris/app.py` (orchestrator run/events endpoints - confirm in verify-first).
- Depends on 20260724-152157: without the turn-start recording, `current` is null
  mid-turn and there is nothing to auto-open.
- Reuse, do not fork: the landing should drive the SAME `createAgentChat` reattach
  seam the per-agent page uses, parameterized by the orchestrator endpoints, rather
  than a parallel streaming path.

## Close-out

Frontend-only; see NOTES.md for the change list + verify-first. Reused the
`createAgentChat` `loadTranscript`/`reattach` seam (no parallel streaming path):
`loadCurrentTranscript` auto-opens the current session on mount and
`reattachOrchestrator` follows the live run via `/api/agents/orchestrator/status`
+ `/events`, injecting the Q1-A prompt. `switchSession` now reuses the shared
`toChatMsgs` mapper. Two new tests (auto-open-idle, live-reattach) plus a local
FakeEventSource harness; the reattach test asserts the mount transcript is fetched
exactly once (no settle re-fetch). Deferred the `onSessionStarted` live-pin (not
in the DoD; `createAgentChat` does not forward that handler - see NOTES.md).

Difficulty: none notable. Formatted only the two touched files (per the
`format-only-the-files-you-edited-not-whole-dirs` lesson), so the diff stayed
clean this time - no revert dance. Used the node_modules symlink (not npm ci) per
the bumped lesson.
