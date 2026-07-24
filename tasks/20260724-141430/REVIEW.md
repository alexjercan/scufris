# Review: Q1-A carry in-flight prompt on run status + inject user bubble on codex reattach

- TASK: 20260724-141430
- BRANCH: fix/reattach-prompt

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

No findings. The out-of-context reviewer diffed the branch against master, read
the surrounding interaction code (reattach/runTurn/settle/ensureBubble;
agent_run_status/_launch_agent_turn/Supervisor.start; strip_steering), ran the
check suite in the worktree, and confirmed each flagged trap holds:

- Injected user bubble renders BEFORE the pending bubble attaches: `runTurn` is
  called with `{ reattach: true }`, so `ensureBubble()` is deferred to the first
  frame; `reattach()` calls `onUserPrompt` (push user msg + render) before
  `subscribeEvents`, so the rebuild includes it and the later-appended pending
  bubble survives. No clobber.
- Dedup guard (last-msg-only) is sound for the live-reattach case: while the run
  is QUEUED/RUNNING the current turn's assistant final answer is not yet in the
  rollout, so a flushed `user_message` is genuinely the last transcript msg;
  otherwise the last msg is a prior assistant reply and it injects. No
  trailing-assistant-after-prompt case exists mid-turn.
- `prompt` exposed only while QUEUED/RUNNING, None otherwise (app.py); backend
  test asserts both live and settled, paired with a real live-state assertion.
- Steering strip mirrors read_transcript exactly (`strip_steering(x).strip()`);
  the stored prompt is the raw/unsteered turn text, so it normalizes to the same
  text the transcript produces.
- No settle-time transcript re-fetch; transcript fetched only at mount.

Check suite (run in worktree): backend `pytest tests/test_app.py` PASS; frontend
`vitest run src/agent-chat-view.test.ts` 32/32 PASS; `npm run lint` clean;
`npm run build` OK; DoD transcript-only-at-mount grep confirmed.

Design: `onUserPrompt` on `StreamHandlers` for a local (non-wire) injection is a
reasonable seam (documented as such; `dispatchStreamEvent` never fires it;
`runTurn`'s handler object is the single funnel for bubble mutations). No
load-bearing choice here needs its own DECISION.md - the status-endpoint-vs-bus
rationale is already in the umbrella GOAL.md.

### In-session supplement (load-bearing re-derivation)

Per the review skill, the in-session pass independently re-derived two
load-bearing claims rather than adopting the round wholesale:

- Steering match: confirmed `STEERING_PREAMBLE` and `AGENT_STEERING_PREAMBLE`
  share the `[scufris-tools]` / `[/scufris-tools]` delimiters and `_STEER_RE`
  matches either, so `strip_steering` strips both preambles - the sub-agent
  (AGENT_STEERING_PREAMBLE) path matches read_transcript too, not just the
  orchestrator path.
- Test teeth: neutering the `reattach` injection makes the "injects the driving
  prompt" test FAIL; independently neutering the last-msg dedup guard makes the
  "does not duplicate" test FAIL. Each new test fails at its OWN boundary, so
  neither is a vacuous pass.

Open `manual:` DoD items (pending user acceptance, batched at flow Finish):
- End-to-end: orchestrator messages a real codex sub-agent; opening that agent's
  chat mid-turn shows the prompt bubble with no reload.
- Diff review: no settle-time transcript re-fetch added.
