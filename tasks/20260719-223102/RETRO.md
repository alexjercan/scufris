# Retro: agent chat - render markdown and code blocks

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- The P0 UX-review gap (unformatted code replies) is closed, and the design call
  paid off: a hand-rolled renderer that NEVER parses untrusted HTML made XSS a
  non-issue rather than a thing to sanitize. Every text run is a `createTextNode`
  and only whitelisted elements are created, so the three XSS pins (raw HTML,
  script-in-fence, `javascript:` link) all pass by construction, not by a filter
  that could be bypassed.
- Zero dependencies was the right fit for this repo's nix + symlinked-worktree
  node_modules setup: no `npm install` into a shared linked dir, no bundle bloat,
  and it matches the plain-TS ethos. A `marked` + `DOMPurify` combo would have
  been more capable but bought complexity the agent's output does not need yet.
- Keeping `renderMarkdown` a pure, side-effect-free function made it exhaustively
  jsdom-testable (10 cases) without touching the chat wiring - the
  `side-effect-free-module-for-jsdom-tests` lesson applied cleanly again.
- Scoping to what the agent actually emits (prose + fenced code + flat lists +
  inline emphasis + links) kept it small; tables/nested lists are an honest,
  documented follow-up rather than a half-built attempt.

## What went wrong / friction

- One real interaction to get right: the assistant bubble had `white-space:
  pre-wrap` (good for plain text), which double-spaces rendered markdown prose.
  Added a `chat__msg--md` modifier that switches white-space back to normal, while
  the code block's own `<pre>` preserves code layout. Easy to miss without
  eyeballing; the split kept both paths correct.

## Lessons

- `build-dom-not-parse-html-for-untrusted-markdown` (frontend/security): to render
  untrusted markdown safely, DON'T parse it to HTML and sanitize - tokenize the
  markdown and BUILD the DOM with `createTextNode` for every text run + a fixed
  element whitelist, and scheme-validate link hrefs. There is then no `innerHTML`
  of model output and thus no XSS surface to filter. Pin it with hostile-input
  jsdom tests (raw HTML, script-in-fence, `javascript:` link). 20260719-223102.

## Follow-ups

- Tables + nested lists if the agent starts emitting them.
- Optional syntax highlighting inside code blocks (the `lang-*` class is already
  set, so a highlighter can hook in without touching the parser).
- Next UX-review P0: live turn progress / streaming (tatr 20260719-223103).
