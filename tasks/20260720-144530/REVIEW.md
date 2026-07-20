# Review: image attachments

- VERDICT: APPROVE
- ROUND: 1

## Summary

Attach an image to a chat turn (file-pick or paste) and codex sees it. The composer
gains an attach button + paste handler + a preview thumbnail; the base64 rides the
`/api/chat/stream` body, is written to a temp file, and passed to codex (`--image`
for exec, `{type:localImage, path}` for app_server). The image renders inline in the
user bubble. 138 pytest + 99 frontend green; and LIVE end-to-end: a red PNG through
the real app_server backend -> codex answered "red".

## What is good

- The cross-cutting signature change (image_paths through the Protocol + 3 impls +
  StreamRunner + both runners + _exec_args) is mechanical and consistent, threaded
  ONLY through the stream path the UI uses (non-stream chat/fork stay text-only), so
  the ripple is minimal. mypy is green across it; the two test doubles were updated.
- Safe by construction on the backend: `_write_image_to_temp` rejects non-image
  mimes and invalid base64, caps at 12MB, and the temp dir is removed in a `finally`
  (so a client disconnect mid-stream still cleans up). A bad attachment yields an
  error frame and never runs the turn - pinned by a test.
- The inline image is the user's OWN data URL (not model output), so setting
  `img.src` is safe; the sanitization story is untouched. The endpoint test proves
  the decoded file actually exists during the turn and is gone after.
- Verified where it matters: a real codex round-trip confirms the localImage/-i path
  actually reaches the model (per probe-runtime-on-target-host-early) - unit tests
  alone could not prove codex "sees" the image.

## Findings

- MINOR (accepted, noted) - v1 scope: ONE image per turn (backend takes a list, so
  multi is a small follow-on); a turn still requires a text message (no image-only);
  and the image is displayed live only - it is not persisted across a reload/switch
  (the transcript re-render has no image). All called out in the task.
- MINOR (accepted) - no client-side size cap; a huge upload is rejected by the 12MB
  backend cap after transfer. Fine for a local single-user app; a pre-check could be
  added later.

## Verdict

APPROVE. A real full-stack feature done carefully: the codex mechanism was probed
first, the signature ripple is contained and typed, the backend is safe (validation
+ temp cleanup), and it is proven end-to-end against live codex. The findings are
explicit v1 trade-offs.
