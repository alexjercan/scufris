# Agent chat: render markdown and code blocks in replies

- STATUS: OPEN
- PRIORITY: 40
- TAGS: feature, agent, ui, spike

## Goal

Assistant replies currently render as raw `textContent` under `white-space:
pre-wrap`, so markdown is dead: code fences show literal backticks, lists show
`- `, tables mangle. For an agent whose job is running CLI tools and returning
code/command output, this is the single biggest daily-use gap. Render assistant
replies as sanitized markdown - fenced code blocks in mono with a copy button,
lists, bold/inline-code, links - while keeping user messages plain.

Hard constraint: the model output is UNTRUSTED, so it must be sanitized before it
goes into the DOM (a real sanitizer, never raw `innerHTML`). Weigh a small vetted
markdown+sanitize dep vs a minimal hand-rolled renderer during /plan (the
nix/webpack build tolerates a dep; judge bundle size vs safety).

## Decision (from /plan): hand-rolled, safe-by-construction renderer

Weighed `marked` + `DOMPurify` vs a small hand-rolled renderer. Chose HAND-ROLLED:

- **Safe by construction.** Instead of parsing untrusted HTML and then sanitizing
  it (marked -> DOMPurify), the renderer NEVER parses HTML: it tokenizes markdown
  and builds the DOM with `createTextNode` for every text run and only a fixed
  whitelist of elements (p, pre, code, ul/ol/li, strong, em, a, h1-3, blockquote).
  There is no innerHTML of model output anywhere, so there is no XSS surface to
  sanitize. Link `href`s are scheme-validated (http/https/mailto/relative only).
- **Zero deps.** No npm install into the symlinked worktree node_modules, no
  bundle bloat; fits the codebase's no-framework, plain-TS ethos.
- **Scoped to what the agent emits.** Covers fenced code (the big one, + copy),
  inline code, bold/italic, ordered/unordered lists, links, headings, paragraphs.
  Tables and nested lists are a NOTED follow-up, not v1.

## Steps

- [ ] `web/src/markdown.ts` (new, side-effect-free): `renderMarkdown(text) ->
      HTMLElement` - a block parser (fenced code / heading / list / blockquote /
      paragraph) + an inline pass (`code`, `**bold**`, `*italic*`, `[t](url)`),
      building DOM with `createTextNode` + whitelisted elements only. Fenced code
      gets a copy button (guard `navigator.clipboard`). Validate link schemes.
- [ ] `web/src/agent-view.ts`: in `renderLog`, render ASSISTANT messages via
      `renderMarkdown` (bubble gets a `chat__msg--md` modifier for normal
      white-space); user/system/pending stay plain `textContent`.
- [ ] `web/src/style.css`: `.md` typography (p / lists / pre+code / inline code /
      links / headings), `.md__copy` button, and `.chat__msg--md { white-space:
      normal }` so prose is not double-spaced while `<pre>` preserves code.
- [ ] `web/src/markdown.test.ts` (jsdom): fenced code -> `pre code` with the code
      text + a copy button; inline code / bold / italic; ordered+unordered lists;
      a safe link (href set) vs a `javascript:` link (rendered inert/as text);
      XSS - a reply with `<img onerror>` / raw HTML produces NO `img`/executable
      markup; plain paragraphs. Plus an `agent-view` test that an assistant reply
      with a code fence renders a `pre` in the log.
- [ ] `npm run ci` green + a serve smoke: an assistant reply with a code block and
      a list renders formatted (user-eyeballed) and the copy button works.

## Definition of Done

- Assistant replies render markdown - fenced code blocks (mono + copy), inline
  code, bold/italic, lists, links, headings - while user/system messages stay
  plain. No innerHTML of model output; hostile markup produces no executable
  nodes (jsdom-pinned). Render side-effect-free for jsdom; `npm run ci` green;
  serve-verified.

## Notes

- Spike: tasks/20260719-223054/SPIKE.md (P0, the worst gap).
- Only assistant messages get markdown; user/system stay plain text.
- Follow-up (noted): tables + nested lists if the agent starts emitting them.
