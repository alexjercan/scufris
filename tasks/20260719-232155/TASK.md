# Chat CSS: restore list markers + polish agent/user message styling

- PRIORITY: 40
- TAGS: feature, agent, ui
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Implementation

- `.md ul { list-style: disc }` / `.md ol { list-style: decimal }` (+ muted
  `::marker`) restore markers stripped by Tailwind Preflight; verified they win in
  the built bundle (higher specificity + later source order).
- Message bubbles: sender-distinguishing asymmetric radius, line-height 1.5 +
  bigger padding, a subtle gradient/shadow on the user bubble, `overflow-wrap:
  anywhere`. Markdown: prose/li line-height, heading hierarchy (h1>h2>h3, h3
  uppercased), link underline-offset + hover. Colours unchanged.
- CSS-only; `npm run ci` green (build + 51 jsdom tests). Visual = user-eyeballed.

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

- [x] Restore markdown list markers: `.md ul { list-style: disc }`,
      `.md ol { list-style: decimal }`, subtle `::marker` colour; keep indent.
- [x] Message bubbles: sender-distinguishing asymmetric radius (a small "tail"),
      slightly larger line-height/padding for readability, keep the user=cyan /
      assistant=panel colours.
- [x] Markdown content: heading size hierarchy (h1 > h2 > h3), readable prose
      line-height, link underline-offset, ensure inline code / code blocks have
      good contrast on the assistant bubble.
- [x] `npm run ci` green (build + existing jsdom tests still pass); user eyeballs
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
