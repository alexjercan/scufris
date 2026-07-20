# Agent chat: image attachments (attach/paste an image to a turn)

- STATUS: OPEN
- PRIORITY: 30
- TAGS: feature,agent,ui

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
