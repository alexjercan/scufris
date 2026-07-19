# Review: chat CSS - restore list markers + polish messages

- DATE: 20260719
- VERDICT: APPROVE (1 round)

## Scope reviewed

`web/src/style.css` only (chat message + `.md` markdown rules). No markup or logic
change.

## Correctness

- Root cause fixed: Tailwind Preflight's `ol, ul, menu { list-style: none }` was
  stripping markdown bullets/numbers. `.md ul { list-style: disc }` /
  `.md ol { list-style: decimal }` restore them. Verified in the BUILT bundle: our
  rules are present and both win over Preflight - higher specificity (`.md ul`,
  0,1,1 vs Preflight's `ul`, 0,0,1) AND later in source order (disc at offset
  14574 > Preflight's none at 5147). `::marker` is muted so the markers do not
  shout.
- The textbox concern was investigated and is a non-issue: Preflight resets form
  controls to `font: inherit; border-radius: 0; background: transparent`, but our
  explicit `.chat__input` / `.chat__editor-input` rules already override all of
  that, so no change was needed there (documented in the task findings).
- Message polish is CSS-only and additive: sender-distinguishing asymmetric radius
  (user squares the bottom-right, assistant the bottom-left), larger line-height
  (1.5) and padding for readability, a subtle gradient + shadow on the user bubble,
  heading size hierarchy (h1 1.15 > h2 1.05 > h3 uppercased 0.96), and link
  underline-offset. The theme colours (user cyan / assistant panel) are unchanged.
- `overflow-wrap: anywhere` added so a long unbreakable token (a URL, a path) wraps
  instead of overflowing the bubble; code blocks still scroll horizontally via
  their own `<pre>`.
- `npm run ci` green: build + 51 jsdom tests still pass (no test touches these
  rules; the render structure is unchanged).

## Nits (non-blocking)

- Visual result is user-eyeballed (no headless browser here), per
  `frontend-verify-needs-e2e-serve`. Structure + bundle are verified; the look is
  the user's call.
- Nested lists still render flat (the renderer does not parse nesting) - unrelated
  to this CSS task; noted in the markdown task's follow-ups.

## Verdict

APPROVE. Markdown lists show markers again (the reported bug), and agent/user
messages read more clearly with distinct bubble shapes, better spacing, and a
heading hierarchy. CSS-only, build-verified; awaiting the user's eyeball.
