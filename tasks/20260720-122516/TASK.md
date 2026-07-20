# Agent chat: file attachments, path references, and rich previews

- STATUS: OPEN
- PRIORITY: 30
- TAGS: feature,agent,ui

## Goal

Make the chat handle files and media, not just text: attach images to a turn,
reference file paths, and render richer previews in replies.

## Split (a /plan decision, 20260720)

The task was a cluster; per the spike ("likely splits at /plan") it is split:
- **Image attachments** -> its own task 20260720-144530 (a large, cross-cutting
  vertical - the codex image-input probe is done and the mechanism captured there).
- **This task now = rich previews (diff rendering)** - the self-contained,
  high-confidence win: a real diff view for ```diff fenced blocks in `markdown.ts`.
- **File-path chips** (click a path to load its content) - later; note only.

## Goal (this task)

Render fenced ```diff code blocks as a real diff view (added/removed/hunk lines
styled) instead of a plain monospace block, so patch/tool output reads visually.

## Notes

- Spike: tasks/20260720-122301/SPIKE.md.
- `markdown.ts` already has `makeCodeBlock(code, lang)`; add a diff-aware branch
  when `lang === "diff"` that builds line rows (`+` added, `-` removed, `@@` hunk,
  context) via createTextNode - NO innerHTML of model output (keep the XSS-free
  DOM-build). Keep the copy button. Pin with jsdom tests incl. a hostile-input
  diff (e.g. a line that looks like HTML).
- Keep render side-effect-free for jsdom; escape everything.
