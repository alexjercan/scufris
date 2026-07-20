# Agent chat: make message affordances always-visible (copy/edit, touch+keyboard)

- STATUS: OPEN
- PRIORITY: 40
- TAGS: feature,agent,ui

## Goal

Copy (assistant reply) and edit (user turn) message actions are `opacity: 0`
until the footer is hovered, so they are invisible on touch and easy to miss with
a mouse (the footer is a thin strip). Make them persistently discoverable - always
visible but dimmed, brightening on hover/focus - and reachable on touch and by
keyboard. A reply-level copy button should be obviously present. Keep the existing
footer (timestamp + action) layout.

## Notes

- Spike: tasks/20260720-102348/SPIKE.md.
- User feedback: "the copy button is not visible unless you hover over it."
- Frontend-only (`agent-view.ts` messageFoot + `style.css` `.chat__copy`/`.chat__edit`).
  Keep render side-effect-free for jsdom; pin the visible-by-default state with a test.
