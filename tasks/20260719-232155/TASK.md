# Chat CSS: restore list markers + polish agent/user message styling

- STATUS: OPEN
- PRIORITY: 40
- TAGS: feature, agent, ui

## Review findings (CSS + Tailwind)

Tailwind v4 is imported (`@import "tailwindcss"`) purely for its Preflight reset +
theme - NO Tailwind utility classes are used in the markup (grep of the HTML found
none). Two Preflight resets affect chat messages:

- `ol, ul, menu { list-style: none }` -> markdown lists render with NO bullets or
  numbers (the user's complaint). Our `.md ul/ol` set `padding-left` (indent) but
  never restored `list-style`, so items look like unindented text.
- Form controls get `font: inherit; border-radius: 0; background: transparent` -
  already overridden by our explicit `.chat__input`/`.chat__editor-input` rules,
  so the textboxes are fine; no change needed there beyond a sanity pass.

## Goal

Improve the agent/user message CSS: restore list markers, and refine the bubble
and markdown-content styling so replies read well.

## Steps

- [ ] Restore markdown list markers: `.md ul { list-style: disc }`,
      `.md ol { list-style: decimal }`, subtle `::marker` colour; keep indent.
- [ ] Message bubbles: sender-distinguishing asymmetric radius (a small "tail"),
      slightly larger line-height/padding for readability, keep the user=cyan /
      assistant=panel colours.
- [ ] Markdown content: heading size hierarchy (h1 > h2 > h3), readable prose
      line-height, link underline-offset, ensure inline code / code blocks have
      good contrast on the assistant bubble.
- [ ] `npm run ci` green (build + existing jsdom tests still pass); user eyeballs
      the visual result.

## Definition of Done

- Markdown lists show bullets/numbers again; agent and user messages look more
  polished (bubble shape, spacing, heading hierarchy, links). No markup/logic
  change - CSS only. `npm run ci` green; user-verified visually.

## Notes

- Direct request (no spike). CSS-only; visual result is user-eyeballed (no
  headless browser), per the `frontend-verify-needs-e2e-serve` lesson.
- The list-style gap is a Tailwind Preflight interaction - document it so the next
  markdown element (tables, if added) remembers Preflight strips defaults.
