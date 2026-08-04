# Agent chat: file attachments, path references, and rich previews

- PRIORITY: 30
- TAGS: feature, agent, ui
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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

## Implementation

- `markdown.ts`: extracted `makeCopyButton(code)` (shared) and added
  `makeDiffBlock(code)` + `diffLineClass(line)`. `makeCodeBlock` dispatches to
  `makeDiffBlock` when `lang === "diff"`. Each diff line becomes a
  `.md__diff-line--{add|del|hunk|meta|ctx}` row with the text set via `textContent`
  (no innerHTML - the XSS-free build is preserved). `diffLineClass` orders the
  file-header markers (`+++`/`---`/`diff `/`index `) BEFORE the `+`/`-` check so
  headers are `meta`, not add/del. Copy button kept.
- `style.css`: `.md__diff` + colored `.md__diff-line--*` (green add, red del, cyan
  hunk, muted meta).

## Tests / verification

- `markdown.test.ts`: a full diff block renders a colored diff view (add/del/hunk
  rows, file headers as meta, no plain `<code>`, copy kept); a hostile `+<img ...>`
  diff line injects no element (textContent). 97 frontend tests green.
- Visuals eyeball-verified in the served bundle (per frontend-verify-needs-e2e-serve).
