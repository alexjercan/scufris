# Q1-A: carry in-flight prompt on run status + inject user bubble on codex reattach

- PRIORITY: 86
- TAGS: bug, agents, frontend, backend, codex, streaming
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As someone watching a codex sub-agent's chat while the orchestrator drives a
turn against it, I want the orchestrator's input prompt to appear as a user
bubble during the live turn, so that the conversation reads from the start
instead of showing only the agent's answer until I reload the page.

Part of umbrella 20260724-141301 (Q1-A). Scope: codex agents.

## Root cause (investigated)

The reattach path streams only the assistant side off the run event bus and
relies on the mount-time transcript load for the user/prompt side
(`web/src/agent-chat-view.ts:452-459`, `:904-919`). Mid-turn, codex has not yet
flushed the `user_message` into the rollout, so `read_transcript`
(`scufris/sessions.py:450`) returns without the prompt and it stays invisible
until the turn ends and a full reload re-reads the rollout. Settle deliberately
does NOT re-fetch the transcript to avoid a session-id write race
(`web/src/agent-chat-view.ts:527-534`) - keep that invariant.

## Approach (A): carry the in-flight prompt on the run status

The run's prompt is known at launch (`_launch_agent_turn`, `scufris/app.py:1156`)
but is not exposed anywhere a reattaching client can read it. Thread it onto the
run state, expose it on `/status` only while live, and have reattach inject the
user bubble before subscribing.

## Steps

- [x] Backend: add `prompt: str | None = None` to `RunState`
      (`scufris/supervisor.py:61`) and mirror it in `snapshot()` (`:113`).
- [x] Backend: add a `prompt` slot to `_Run` (`scufris/supervisor.py:72`), accept
      it in `_Run.__init__` (`:89`), and add a keyword `prompt: str | None = None`
      to `Supervisor.start` (`:181`), passing it into the `_Run(...)` construction
      (`:208`).
- [x] Backend: pass `prompt=prompt` from `_launch_agent_turn`'s `supervisor.start`
      call (`scufris/app.py:1231`). Steering is added DOWNSTREAM inside the codex
      turn path (`_steer` at `scufris/agent.py:583`), so the prompt captured here is
      already the raw, unsteered user text; store it as-is.
- [x] Backend: add `prompt: str | None = None` to `AgentRunStatus`
      (`scufris/app.py:500`). In `agent_run_status` (`:1311`), set
      `result.prompt = strip_steering(run_state.prompt).strip() or None` ONLY when
      `run_state` is not None and its state is queued/running (import/confirm
      `strip_steering` from `scufris/sessions.py:87` is available in app.py). Leave
      it None for idle/finished runs.
- [x] Frontend types: add `prompt?: string | null` to the `AgentRunStatus`
      interface in `web/src/common.ts`.
- [x] Frontend handler seam: add optional `onUserPrompt?(text: string): void` to
      `StreamHandlers` in `web/src/chat-stream.ts` (no dispatch wiring needed - it
      is not a wire event, only a local injection hook).
- [x] Frontend runTurn: implement `onUserPrompt` in `runTurn`
      (`web/src/agent-chat-view.ts:460`+) to push `{ role: "user", text, ts:
      Date.now() }` into `msgs` and call `render()`. It must run BEFORE
      `ensureBubble()` attaches the pending bubble (render() rebuilds `log` from
      `msgs` and would wipe a pre-attached bubble; in reattach mode `ensureBubble`
      is deferred to the first frame, so calling `onUserPrompt` at the start of the
      reattach runner is safe). Guard duplication: skip if the last `msgs` entry is
      already a `role: "user"` message whose text equals the injected text.
- [x] Frontend reattach: in `startAgentChat`'s `reattach`
      (`web/src/agent-chat-view.ts:904-919`), after the running/queued gate and
      before `subscribeEvents`, if `status.prompt` is set call
      `handlers.onUserPrompt?.(status.prompt)`.
- [x] Backend test: extend `tests/test_app.py` - drive a live run for an agent and
      assert `/api/agents/{id}/status` returns `prompt` (steering-stripped) while
      queued/running, and `prompt is None` once idle/finished. Use the existing
      status-test harness patterns (e.g. around `tests/test_app.py:2677`, the
      `builder` agent live-run fixtures).
- [x] Frontend test: extend `web/src/agent-chat-view.test.ts` - a reattach whose
      `/status` returns a `prompt` injects a user bubble before the streamed
      assistant content; and a reattach whose transcript already ended with that
      prompt does NOT duplicate it.
- [x] Run the full gate: `nix flake check` (ruff + mypy + pytest) and `npm test`
      in `web/`; both green.

## Definition of Done

- `/api/agents/{id}/status` exposes the in-flight turn's prompt (steering
  stripped) while the run is queued/running, and null when idle/finished
  (test: a new `tests/test_app.py` case asserting both).
- On live reattach, the codex sub-agent chat renders the orchestrator's prompt
  as a user bubble before the assistant stream, and does not duplicate it when
  the transcript already ends with that prompt (test: new
  `web/src/agent-chat-view.test.ts` cases for inject + no-duplicate).
- No settle-time transcript re-fetch is introduced (cmd: `grep -n "loadTranscript\|transcript" web/src/agent-chat-view.ts` shows the fetch only at mount, not in `settle`).
- Full QA gate is green (cmd: `nix flake check`) and the web suite is green
  (cmd: `cd web && npm test`).
- manual: orchestrator messages a real codex sub-agent; opening that agent's
  chat mid-turn shows the prompt bubble with no reload.

## Notes

- Key files: `scufris/supervisor.py` (RunState/_Run/start), `scufris/app.py`
  (`AgentRunStatus:500`, `_launch_agent_turn:1153`, `agent_run_status:1311`),
  `scufris/sessions.py` (`strip_steering:87`), `web/src/common.ts`
  (AgentRunStatus type), `web/src/chat-stream.ts` (StreamHandlers),
  `web/src/agent-chat-view.ts` (runTurn:460, reattach:904).
- `strip_steering` + `.strip()` mirrors `read_transcript`'s user-message
  handling (`scufris/sessions.py:482`) so the live bubble matches the
  post-reload transcript exactly and the duplication guard works.

## Close-out (what changed, why, difficulties, reflection)

Implemented Approach A end to end:

- Backend: `RunState` (`scufris/supervisor.py`) gained a `prompt` field, mirrored
  through `_Run` (slot + ctor kwarg) and `snapshot()`; `Supervisor.start` gained a
  `prompt` kwarg. `_launch_agent_turn` (`scufris/app.py`) passes the turn prompt in.
  `AgentRunStatus` gained a `prompt` field, set in `agent_run_status` only when the
  run is QUEUED/RUNNING, via `strip_steering(run_state.prompt).strip() or None`.
- Frontend: `AgentRunStatus` type (`web/src/common.ts`) and a local-only
  `onUserPrompt` hook on `StreamHandlers` (`web/src/chat-stream.ts`, NOT wired into
  `dispatchStreamEvent` - it is an injection hook, not a wire event). `runTurn`
  implements it (push a user `ChatMsg` + `render()`, deduped against the last msg);
  `startAgentChat`'s `reattach` calls it with `status.prompt` after the live gate,
  before `subscribeEvents`.
- Tests: `tests/test_app.py::test_status_exposes_in_flight_prompt_stripped` drives a
  blocking mock turn and asserts `/status.prompt` is the steering-stripped text
  while live and None once settled. `web/src/agent-chat-view.test.ts` adds the
  inject-when-transcript-lacks-it and no-duplicate-when-transcript-has-it cases
  against the real `startAgentChat` wiring (fetch stub + FakeEventSource).

Difficulty / correction: the plan assumed the captured prompt still carried the
steering preamble. Reading the code showed `_steer` runs DOWNSTREAM in the codex
turn path (`scufris/agent.py:583`), so the stored prompt is already unsteered and
already equals `read_transcript`'s stripped output - the dedup guard works without
the strip. Kept `strip_steering().strip()` at the status boundary anyway as a
belt-and-suspenders match to the transcript's exact transform (and exercised it in
the test by sending a pre-steered message through the real chat path). Step 3 text
was corrected to match reality.

Reflection: grounding the plan against the actual `_steer` call site before writing
would have avoided the inaccurate step. The status endpoint (already GET-ed by
reattach purely to gate on live-state) was the right carrier - no new bus event,
no replay-buffer concerns. `manual:` end-to-end check with a real codex sub-agent
is deferred to the flow Finish gate.
- Why the status endpoint and not a new bus event: `reattach` already GETs
  `/status` solely to gate on running/queued, so it is the natural carrier and
  avoids event-bus replay-buffer eviction concerns. (Approach B - a leading
  prompt StreamEvent - was considered and set aside.)
- Do NOT re-fetch the transcript on settle; the race note at
  `web/src/agent-chat-view.ts:527-534` documents why.
- The prompt stored on RunState is the RAW turn prompt; steering is stripped at
  the read boundary (status endpoint), matching read_transcript's boundary.
