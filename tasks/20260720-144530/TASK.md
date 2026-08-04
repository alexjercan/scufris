# Agent chat: image attachments (attach/paste an image to a turn)

- PRIORITY: 30
- TAGS: feature, agent, ui
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Attach an image to a chat turn (paste or file-pick) so the agent can see it. codex
supports this natively on BOTH backends - this task wires it end to end.

## Probe findings (de-risked, 20260720)

- exec: `codex exec -i/--image <FILE>...` attaches image files to the prompt.
- app_server: `turn/start` `input` is `Array<UserInput>` where UserInput includes
  `{ "type": "localImage", detail?, path }` (from `codex app-server generate-ts`
  -> v2/UserInput.ts). So a local image file PATH works on both backends.
  (UserInput also has `skill` and `mention` variants - future.)

## Shape

- Transport: the composer reads the image as base64; POST /api/chat/stream body
  gains an optional `image {data_base64, mime}` (JSON, next to `message`). v1 = ONE
  image per turn.
- Backend: decode -> a temp file (cleaned up after the turn); thread an
  `image_path` through the agent. This is a cross-cutting signature change: the
  `Agent.chat_stream` Protocol + DisabledAgent/MockAgent/CodexCliAgent +
  `_run_codex_exec`/`_stream_codex_exec`/`_stream_app_server`/`_exec_args` all take
  an optional image. exec -> add `-i <path>`; app_server -> add
  `{type:localImage, path}` to the turn input array.
- Frontend: an attach button (+ paste handler) in the composer, a thumbnail chip
  before send, include base64 in sendChatStream, clear after; render the image
  inline in the user bubble (live only - persistence across reload is out of scope
  for v1; note it).
- Verify LIVE that a turn with an attached image is understood (both backends).
- Escape everything; keep render side-effect-free for jsdom; gate on agent enabled.

## Implementation

- Backend (`agent.py`): threaded `image_paths: list[str] | None` through the STREAM
  path only (the UI path) - `Agent.chat_stream` Protocol + DisabledAgent/MockAgent/
  CodexCliAgent, the `StreamRunner` type, `_stream_codex_exec`, `_stream_app_server`,
  `_exec_args`. exec adds `--image <path>` per image; app_server appends
  `{type:localImage, path}` items to the `turn/start` input array. Non-stream
  `chat`/fork left text-only.
- Backend (`app.py`): `ImageAttachment {data_base64, mime}` on `ChatRequest`;
  `_write_image_to_temp` decodes (validate base64, reject non-image mime, cap at
  12MB) to a temp dir; `post_chat_stream` writes it before the turn, passes
  `image_paths`, and removes the temp dir in a `finally`. A bad attachment yields an
  error SSE frame and never runs the turn.
- Frontend: composer gets an attach button (opens a file picker) + a textarea paste
  handler; a floating preview thumbnail with a remove (x); `_pendingImage` state.
  `sendChatStream` includes `image` in the body only when attached; `submit`/
  `runStreamingTurn` thread it; the user `LogEntry.imageUrl` renders the image inline
  in the bubble (the user's own data URL, not model output). v1: ONE image/turn,
  live display only (not persisted across reload - noted).

## Verification

- Backend: `_exec_args` adds `--image`; the endpoint writes a real temp file that
  exists during the turn and is cleaned up after; a non-image mime is rejected with
  an error frame and no turn. 138 pytest.
- Frontend: `sendChatStream` includes/omits the image in the POST body; the full
  attach->preview->send flow renders the image in the user bubble and clears the
  preview. 99 frontend tests.
- LIVE e2e (real app_server backend + codex): posted a 48x48 red PNG asking for the
  dominant color -> codex replied "red". The image pipeline is understood end to end.
