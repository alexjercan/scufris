# Agent chat: file attachments, path references, and rich previews

- STATUS: OPEN
- PRIORITY: 30
- TAGS: feature,agent,ui

## Goal

Make the chat handle files and media, not just text: attach images to a turn,
reference file paths, and render richer previews in replies.

## Notes

- Spike: tasks/20260720-122301/SPIKE.md.
- User: "maybe file attachments; file paths; preview things".
- codex natively attaches images: `codex exec -i/--image <FILE>...`; the
  app-server `turn/start` input is an array that can carry image items (PROBE the
  exact shape before committing a backend - open question in the spike).
- Likely splits at /plan. Cheapest high-value wins to lead with:
  (a) image attachment in the composer -> pass to codex -i / app-server input, and
      render the attached image inline in the user bubble;
  (b) a real diff view for ```diff fenced blocks in `markdown.ts` (a second code
      hook), so tool/patch output reads visually.
  Later: file-path chips (click to load a file's content as context), and previews
  for rendered files / command output.
- Untrusted model output stays sanitized (build DOM, no innerHTML). Keep render
  side-effect-free for jsdom.
