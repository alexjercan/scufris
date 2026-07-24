# Goal: show the orchestrator prompt in a codex sub-agent chat during live reattach (Q1-A)

- DATE: 20260724
- UMBRELLA TASK: 20260724-141301
- LANDING SCOPE: squash-merge each task to local `master`; do NOT push (user's call)

## Goal

When the orchestrator drives a turn against a codex sub-agent and you open (or
are already viewing) that sub-agent's chat mid-turn, the orchestrator's input
prompt is invisible: the reattach path streams only the assistant side off the
run event bus and relies on the mount-time transcript load for the user/prompt
side, but codex has not yet flushed the `user_message` into the rollout, so
`read_transcript` returns without it. The prompt only appears after the turn
ends and a full page reload.

Fix it with Approach A: carry the in-flight turn's prompt on the run status.
`reattach()` already GETs `/api/agents/{id}/status` and gates on
running/queued; thread the current turn's prompt through the run state so the
status endpoint can expose it, and have `reattach()` inject a user bubble
(steering stripped, matching `read_transcript`) before subscribing - unless the
transcript already ended with that prompt. Do NOT re-fetch the transcript on
settle (that reintroduces the documented session-id write race,
`web/src/agent-chat-view.ts:527-534`).

## Done means

1. `AgentRunStatus` exposes the in-flight turn's prompt (steering-stripped) while
   the run is queued/running, and null/absent otherwise (test: backend test on
   `/api/agents/{id}/status` for a live run).
2. On live reattach, the sub-agent chat renders the orchestrator's prompt as a
   user bubble before the assistant stream, and does not duplicate it when the
   transcript already ended with that prompt (test: agent-chat-view frontend
   test for the reattach-injects-prompt and no-duplicate cases).
3. No transcript re-fetch is added on settle; the session-id write race note
   stays honored (manual: confirm the diff adds no settle-time transcript fetch).
4. The prompt shows live end to end for a real codex sub-agent turn driven by
   the orchestrator (manual: orchestrator messages a codex sub-agent, open that
   agent's chat mid-turn, confirm the prompt bubble appears without a reload).

Overall: `nix flake check` (ruff + mypy + pytest) green, and the web test suite
(`npm test` in `web/`) green.

## Tasks

Updated as tasks land (one line per land).

- [x] 20260724-141430 (p86, scufris) Q1-A: carry in-flight prompt on run status + inject user bubble on codex reattach
      landed fa05fb9; 1 review round (out-of-context APPROVE, zero findings); lesson `plan-locates-transform-from-the-call-site-not-the-model` added

## Decisions (load-bearing, architectural)

- (none yet)

## Manual acceptance (batched for the user at Finish)

- (PENDING USER) end-to-end: orchestrator -> codex sub-agent, open chat mid-turn,
  prompt bubble appears with no reload (done-def item 4). Needs a live codex
  agent; cannot be exercised from the harness. Left pending your acceptance.
- (confirmed by grep at Finish) diff review: no settle-time transcript re-fetch
  added - `/transcript` is fetched only at mount (agent-chat-view.ts:908, called
  from :857), never in `settle`. Done-def item 3.

## Finish

- Done-def items 1-3 proven by tests + grep; overall green bar met (nix flake
  check + web CI green on master fa05fb9). Item 4 is a live-codex manual check,
  left pending user acceptance above.
- Conformance clean (`tatr check --ledger LESSONS.md` exit 0). No loose scratch
  (`/lessons` no-op). One lesson added at x1 (not promotion-eligible).
- No residue: Q2 remains parked as task 20260724-141150 (p0) for offline
  prioritization; no other deferred items.
