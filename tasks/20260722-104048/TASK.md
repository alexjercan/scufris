# broad styling pass over all pages - sharper terminal aesthetic from kitty config

- STATUS: OPEN
- PRIORITY: 30
- TAGS: frontend,css,styling

## Goal

A broad styling pass over ALL pages (CSS + HTML): cleaner, sharper, more
consistent components - especially the buttons (the `/agents` page buttons are
called out as weak). Push the terminal aesthetic further. The user likes the
existing color scheme and the landing page, so keep the palette and refine, do
not redesign.

## Why

User feedback (2026-07-22): "the buttons (especially in /agents/) are not that
great, I would do a big review over all the pages + CSS + HTML and improve the
styling... cleaner and sharper, terminal style... check my kitty config + other
terminal related colors and style preferences to make it like that".

## Notes / scope to pin

- Read the user's kitty config (~/.config/kitty/ likely) and other terminal color
  prefs and align the web palette/typography to them where it improves the
  terminal feel. Keep the current scheme's character.
- Audit buttons/cards/forms/nav across every page for consistency; unify a button
  component; sharpen spacing/borders.
- Likely a /spike (inventory the current CSS + capture the kitty palette) then a
  staged implementation, page by page.
- LOW-PRIORITY sibling idea (ideation only, do NOT implement until much later):
  user-configurable styling/theming. Capture as a separate ideation task.
