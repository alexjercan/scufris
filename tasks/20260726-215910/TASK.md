# Persist codex 'thinking' reasoning across a page reload (backend sidecar)

- PRIORITY: 0
- TAGS: feature, agents, frontend, backend, codex, streaming
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As someone reading a codex agent's chat, I want a past assistant turn's
"thinking" (reasoning) spoiler to STILL be present after a full page reload, so I
can expand the reasoning of earlier turns in a session I have come back to, not
just the live/just-settled turn.

This is the INVOLVED half of the original combined task (`20260724-141150`, now
split). It builds on the ephemeral task `20260726-215847` (in-session
persistence) and adds true cross-reload persistence, which requires a backend
sidecar because reasoning cannot be recovered from disk (see Constraint).

## Constraint (verified against a real rollout - this is why it is backend work)

Reasoning is NOT recoverable from codex's own on-disk transcript. Codex persists
reasoning as a `response_item` of `type: "reasoning"` whose `summary` is `[]` and
whose body is an `encrypted_content` blob (checked
`~/.codex/sessions/.../rollout-*.jsonl`). The plaintext thinking only exists in
the live `reasoning_delta` stream (`scufris/agent.py:88` `StreamReasoningDelta`).
So `read_transcript` (`scufris/sessions.py:450-508`, which today keeps only
`user_message` + the `agent_message` with `phase == final_answer` and explicitly
skips reasoning phases) cannot reconstruct it. If we want reload-survival,
Scufris must capture the live stream itself and store it in a sidecar.

## Steps

- [x] Backend: as a turn streams, capture the `reasoning_delta` text per
      (session, turn) and persist it in a sidecar (e.g. alongside the session
      transcript). Decide the storage shape and keying.
      -> `scufris/reasoning_store.py` (per-session file under state_dir);
      captured in `app.py` `turn_stream()` on StreamDone. See DECISION.md.
- [x] Extend `TranscriptMessage` (`scufris/sessions.py`, `web/src/common.ts`)
      with an optional `reasoning` field.
- [x] Merge the sidecar reasoning into the `/transcript` response so a reloaded
      transcript carries reasoning on the relevant assistant messages.
      -> `CodexBackend.read_transcript` + pure `sessions.merge_reasoning`
      (tail-align + fingerprint guard).
- [x] Frontend: in `renderChatLog`, render a transcript message's `reasoning` as
      the same hidden-by-default `<details class="chat__thinking">` spoiler used
      live and by the ephemeral task - so live, settled, and reloaded turns all
      look identical.
      -> `renderChatLog` already rendered `reasoning`; wired the two
      transcript->ChatMsg mappings (agent-chat-view.ts, agent-view.ts) to pass it.

## Definition of Done

- After a FULL page reload, past assistant turns show the same collapsed
  "thinking" spoiler, expandable to the reasoning that streamed during that turn
  (manual: send a codex turn, reload the page, expand thinking on the past turn).
- The spoiler is collapsed by default on reload (manual: confirm hidden/closed on
  first render).
- Existing sessions without a sidecar degrade gracefully: no reasoning shown, no
  error (manual: open a pre-existing session and confirm it renders normally).

## Design (accepted - see DECISION.md)

- Sidecar store: a new `scufris/reasoning_store.py` (mirrors `agent_store`
  conventions) keyed by codex `session_id`, ONE JSON file per session under
  `state_dir/reasoning/<session_id>.json` (per-session file, not one shared
  file, so a turn's append is O(1) rather than rewriting every session's
  reasoning). Shape: `{"turns": [{"answer": "<fingerprint>", "reasoning":
  "..."}, ...]}`, ordered oldest->newest, one entry per COMPLETED assistant
  turn (empty `reasoning` when the model did not think, to keep 1:1 with the
  assistant messages). `session_id` is charset-guarded before use as a
  filename.
- Capture: in `_launch_agent_turn.turn_stream()` (`app.py`), accumulate
  `StreamReasoningDelta.delta` and, on `StreamDone` with non-empty
  `reply.text`, append one entry keyed by the captured `session_id`, BEFORE
  the done frame is yielded to the bus (so a reload triggered by the done
  frame reads a sidecar that is already written). Gated to the codex backend.
- Merge: `CodexBackend.read_transcript` loads the sidecar and tail-aligns it
  with the assistant messages via a pure helper in `sessions.py`; each pair is
  guarded by a normalized text fingerprint so a partial/pre-existing sidecar
  (feature deployed mid-session) attaches reasoning only to the turns it
  genuinely covers and degrades to nothing on mismatch (no error).
- `sessions.read_transcript` stays pure (codex_home only); the sidecar lives in
  `state_dir`, so the merge happens at the backend boundary.

## Notes

- Split out of the original combined task `20260724-141150` on 2026-07-26.
- Depends on: 20260726-215847 (ephemeral in-session persistence). That task
  establishes the settled-message `reasoning` shape and the `renderChatLog`
  spoiler rendering this task reuses for reloaded transcripts.
- Codex-only for now; other backends not tested for reasoning streaming.
- Key files: `scufris/agent.py` (StreamReasoningDelta, _appserver_event),
  `scufris/sessions.py` (read_transcript, TranscriptMessage, the `/transcript`
  path), `web/src/agent-chat-view.ts` (renderChatLog), `web/src/common.ts`
  (TranscriptMessage type).
