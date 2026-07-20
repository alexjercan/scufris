# Review: diff-block rich preview

- VERDICT: APPROVE
- ROUND: 1

## Summary

```diff fenced blocks now render as a colored diff view (add/del/hunk/meta rows)
instead of a plain monospace block, so patch/tool output reads visually. Split out
of the 122516 cluster (image attachments -> its own task 20260720-144530, probed +
specified). 97 frontend tests green.

## What is good

- Preserves the XSS-free invariant the markdown renderer is built on: each diff
  line's text is set via `textContent`, never innerHTML - pinned by a hostile
  `+<img ...>` test that injects no element. This was the one real risk and it's
  covered.
- `diffLineClass` gets the ordering right (file-header `+++`/`---`/`diff `/`index `
  BEFORE the `+`/`-` add/remove check), so headers are `meta` not add/del - the
  classic diff-highlighting bug, and it's tested.
- Clean refactor: the copy button is extracted to `makeCopyButton` and shared, so
  the diff block keeps copy with no duplication; `makeCodeBlock` just dispatches on
  `lang === "diff"` (case-insensitive).
- The diff `<pre>` reuses the existing `.md__code pre` chrome (bg/border/overflow-x)
  so long lines scroll rather than wrap, consistent with normal code blocks.

## Findings

- MINOR (accepted) - a trailing newline in a diff would yield one empty `ctx` row;
  harmless (renderMarkdown joins fence lines without a trailing newline anyway).
- NIT - intraline (word-level) diff highlighting is not done; line-level is the
  right scope for reading agent/patch output. Not needed.

## Verdict

APPROVE. Small, correct, keeps the sanitization guarantee, and tested including the
hostile case and the header-ordering edge. Visuals eyeballed per the frontend lesson.
