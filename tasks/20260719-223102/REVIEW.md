# Review: agent chat - render markdown and code blocks in replies

- DATE: 20260719
- VERDICT: APPROVE (1 round)

## Scope reviewed

`web/src/markdown.ts` (new, `renderMarkdown`), `web/src/markdown.test.ts` (new),
`web/src/agent-view.ts` (assistant messages -> markdown), `web/src/style.css`
(`.md*`), `web/src/agent-view.test.ts` (one integration test).

## Correctness

- Safe by construction, verified: every text run is a `createTextNode` and only a
  fixed element whitelist is created - there is no `innerHTML` of model output.
  Three XSS pins prove it: raw `<img onerror>` survives as literal text (no `img`
  node), `<script>` inside a fence becomes `code.textContent` (no `script`), and a
  `[click](javascript:alert(1))` link renders inert as plain text (no `a`). Link
  hrefs are scheme-validated (`https?`/`mailto`/relative/anchor only).
- Rendering covers what the agent emits, all tested: fenced code -> `pre code`
  with the raw code + a `lang-*` class + a copy button; inline code / bold /
  italic; ordered + unordered lists; safe links (href + `rel=noopener`); headings;
  and plain prose as `<p>` paragraphs. The copy button guards `navigator.clipboard`
  (absent in jsdom / on insecure origins), so it no-ops rather than throwing.
- Integration is correctly scoped: only ASSISTANT messages go through
  `renderMarkdown` (with a `chat__msg--md` modifier that switches off the
  plain-text `pre-wrap` so prose is not double-spaced, while the code block's own
  `<pre>` preserves layout). User/system/pending stay `textContent` - pinned by a
  test that a user message containing triple-backticks renders NO `pre`.
- Module is side-effect-free (no import-time work), so it unit-tests cleanly under
  jsdom (the `side-effect-free-module-for-jsdom-tests` lesson).
- `npm run ci` green: 51 jsdom tests (4 files) + format + lint + build. Bundle
  ships `renderMarkdown`/`md__code`/`md__copy`.

## Nits (non-blocking)

- Tables and nested lists are out of scope (rendered as plain paragraphs) -
  a documented follow-up, per the plan. The agent's typical output (prose + code +
  flat lists) is covered.
- The inline pass is non-nested (e.g. bold-inside-a-link is not merged); fine for
  replies and never unsafe (worst case is under-formatting, never injection).
- No syntax highlighting inside code blocks; the `lang-*` class is set so a
  highlighter could be added later without touching the parser.

## Verdict

APPROVE. The single biggest daily-use gap is closed: assistant replies now render
markdown with copyable code blocks, built with zero dependencies and no XSS
surface (no untrusted HTML is ever parsed). User messages stay plain; hostile
output is inert. jsdom-pinned, `npm run ci` green.
