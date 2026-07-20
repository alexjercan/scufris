# Retro: image attachments

- DATE: 20260720
- TASK: 20260720-144530
- LANDED: 0c75cde

## What shipped

Attach an image to a chat turn (file-pick or paste) and the model sees it.
base64 rides the `/api/chat/stream` body -> decoded to a temp file (mime +
base64 validated, 12MB cap, cleaned up in a `finally`) -> passed to codex as
`--image <path>` (exec) or a `{type:localImage, path}` input item (app_server).
The composer got an attach button + paste handler + preview thumbnail; the image
renders inline in the user bubble. 138 pytest + 99 frontend green; a live red-PNG
round-trip through the real app_server backend returned "red".

## What went well

- Probing the codex contract statically FIRST (`codex app-server generate-ts` ->
  localImage; `codex exec --help` -> -i/--image) meant the cross-cutting signature
  change was written once against a known shape, not guessed and reworked. Same
  lesson as [[probe-runtime-on-target-host-early]] but applied at design time.
- Threading the image through the STREAM path only (not non-stream chat/fork) kept
  the Protocol ripple small while still covering the one path the UI uses. The
  temptation to "do it everywhere for symmetry" would have doubled the surface for
  no user-visible gain.
- The temp-file lifecycle (create before the turn, rmtree in `finally`) was pinned
  by a test that asserts the file EXISTS mid-turn and is gone after - a lifecycle
  claim that a "does it decode" test would have missed.

## What was tricky

- Two test doubles broke on the signature change (stream_runner arity, FakeAgent
  kwarg). Expected fallout of a Protocol change; the fix is mechanical but the
  doubles are easy to forget because they are not the thing under test. Grepping
  for every implementor of the changed Protocol before running is faster than
  discovering them one failure at a time.
- The error SSE frame uses `json.dumps` (spaces after colons) while event frames use
  compact `model_dump_json`; a test asserting `'"kind":"error"'` failed on the space.
  Re-confirmed [[error-frames-use-json-dumps-not-model-dump-json]] - assert on the
  actual serializer's output, not the compact form.
- The worktree carried an untracked `web/node_modules` (symlinked build dir), so
  `sprout rm` refused and left the dir after deleting the branch. Had to
  `rm -rf` the leftover. Minor, but worth noting: sprout rm is not force by default.

## Lessons for next time

- For a feature that depends on an external tool's wire contract, spend the first
  five minutes making the tool tell you its schema (generate-ts, --help, a probe
  turn) rather than reading its source or guessing. It collapses the
  design-rework loop.
- When changing a Protocol signature, grep for every implementor AND every test
  double up front and change them in one pass; the doubles are the ones that bite.
- A capability like "the model can see an image" is only proven by a live
  round-trip, never by unit tests - budget the e2e as part of the task, not as an
  optional extra.
