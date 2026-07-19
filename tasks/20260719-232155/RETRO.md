# Retro: chat CSS - restore list markers + polish messages

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- Root-caused the missing bullets to Tailwind v4 Preflight (`ol, ul, menu {
  list-style: none }`) by grepping the BUILT bundle, not guessing - and confirmed
  the same way that the fix wins (higher specificity + later source order). The
  user's instinct ("probably comes from tailwind") was exactly right.
- Also confirmed the textbox concern was already handled: Preflight resets form
  controls, but our explicit `.chat__input`/`.chat__editor-input` rules override
  it - so the review honestly reported "no change needed there" instead of
  inventing work.
- CSS-only, additive polish (bubble shape/spacing, heading hierarchy, links) kept
  the change low-risk: `npm run ci` stayed green because no render structure moved.

## What went wrong / friction

- Nothing notable. The one honest limit is that the visual result can only be
  user-eyeballed here (no headless browser), so the review verified structure +
  bundle and handed the look to the user.

## Lessons

- `tailwind-preflight-strips-defaults` (frontend/css): Tailwind's Preflight base
  reset removes user-agent defaults - notably `list-style: none` on ul/ol and
  native form-control styling - so anything rendered as real markdown/HTML needs
  its defaults restored explicitly (`.md ul { list-style: disc }`). When a styled
  element looks "unstyled", grep the BUILT bundle for the Preflight rule before
  guessing. 20260719-232155.

## Follow-ups

- If the markdown renderer gains tables (a noted follow-up), remember Preflight
  also flattens table borders/spacing - they will need explicit styling too.
- Next UX-review P0 still open: live turn progress / streaming (tatr 20260719-223103).
