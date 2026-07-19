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

## Notes

- Spike: tasks/20260719-223054/SPIKE.md (P0, the worst gap).
- Keep the render side-effect-free for jsdom; add a jsdom test that a hostile
  reply (e.g. `<img onerror>`, `javascript:` link) produces no executable markup.
- Only assistant messages get markdown; user/system stay plain text.
