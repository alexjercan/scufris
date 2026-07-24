# Goal: orchestrator (codex) session available at send, not only after the turn finishes

- DATE: 20260724
- UMBRELLA TASK: 20260724-151911
- LANDING SCOPE: squash-merge each task to local `master`; do NOT push (user's call)

## Goal

On the orchestrator chat (codex backend), sending a message and then refreshing
the page mid-turn shows NO session in the switcher until the turn finishes - the
session only appears once the conversation is done. Root cause: scufris records
the session id in its registry only in the terminal `persist` callback ->
`mark_finished` (`scufris/app.py:1199`, `scufris/agent_store.py:705`), so
mid-turn `orchestrator_session_id()` still returns the pre-turn value (None for a
fresh chat).

The codex thread/session id is already known EARLY - right after
`thread/start` / `thread/resume` returns (`scufris/agent.py:573-575`), before the
turn streams - it is just not surfaced until the final `StreamDone`. Surface it
at that point and record ownership in the registry immediately, so a mid-turn
refresh finds the session (and, with the just-landed Q1-A change, the prompt
bubble + live reply).

Scope: codex only (claude/opencode already record ownership at launch on the
provider side; their registry timing is out of scope for this goal). This is the
deferred codex half of the launch-time session-ownership line of work (part 2,
task 20260724-111955, deferred codex explicitly).

## Done means

1. On a fresh codex orchestrator turn, the session id is recorded in the registry
   as the current session at turn-start (as soon as `thread/start` returns), not
   at `mark_finished` (test: a backend/app test asserting `orchestrator_session_id()`
   / `/api/agent/sessions` `current` is set WHILE the run is live, before the
   terminal frame).
2. `/api/agent/sessions` lists the in-flight session (not filtered out for a
   just-created rollout) and marks it current mid-turn (test: endpoint test with a
   live run).
3. After sending on the orchestrator landing and refreshing mid-turn, the session
   appears in the switcher and the in-flight turn is reflected (streaming reply +
   prompt bubble), with no reload-after-finish needed (manual: real codex turn).
4. A turn that starts its thread but then errors still leaves a correctly-recorded
   session (no regression vs today's mark_finished-on-error) (test).

Overall: `nix flake check` (ruff + mypy + pytest) green and `web` `npm run ci` green.

## Tasks

Updated as tasks land (one line per land, in intended order).

- [x] 20260724-152157 (p85, scufris) Record codex session in the registry at turn-start (early StreamSessionStarted -> set_current)
      landed f698273; 1 review round (out-of-context APPROVE, zero findings); DECISION.md added; lessons: format-only-touched-files (x2), symlink-node_modules (x3)
- [x] 20260724-152230 (p83, scufris) Reflect the in-flight orchestrator session on the landing after refresh (auto-open current + reattach) [depends on 20260724-152157]
      landed 02bc055; 1 review round (out-of-context APPROVE; 1 MINOR WONTFIX + 1 NIT fixed); onSessionStarted live-pin deferred (NOTES.md)

## Decisions (load-bearing, architectural)

- 20260724-152157 DECISION.md: surface the codex session id at turn-start via a
  StreamSessionStarted event (not a callback / not a status-poll) (ACCEPTED)

## Manual acceptance (batched for the user at Finish)

- (PENDING USER) end-to-end: send on the codex orchestrator chat, refresh mid-turn
  -> session shows in the switcher AND the live turn is reflected (streaming reply
  + prompt bubble) without waiting for the turn to finish (done-def item 3). Needs
  a live codex agent; cannot be exercised from the harness.

## Finish

- Done-def items 1, 2, 4 delivered by task 20260724-152157 (landed f698273);
  item 3 delivered by task 20260724-152230 (landed 02bc055). Both proven by tests;
  the end-to-end refresh behavior is the one manual check above.
- Overall green bar met: `nix flake check` + `web` `npm run ci` green on master
  02bc055.
- Conformance clean (`tatr check --ledger LESSONS.md` exit 0). No loose scratch
  (`/lessons` no-op). Lessons this goal: `format-only-the-files-you-edited-not-whole-dirs`
  (x2, new), `symlink-node_modules-into-fresh-worktrees` (bumped to x3, GUARDED).
- Residue / deferred: the `onSessionStarted` live-pin on the landing (fork-during
  a fresh unsettled turn, a pre-existing edge case) is deferred with a written
  rationale in task 20260724-152230 NOTES.md - not filed as its own task (low
  value, out of goal scope). Nothing else outstanding.
