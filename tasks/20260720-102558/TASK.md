# Agent chat: make message affordances always-visible (copy/edit, touch+keyboard)

- PRIORITY: 40
- TAGS: feature, agent, ui
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

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

## Implementation

- `style.css`: `.chat__copy, .chat__edit` resting `opacity: 0` -> `0.6` (always
  visible but dimmed), and the brighten rule adds `:hover` on the buttons
  themselves (was only `.chat__foot:hover` + `:focus`), so opacity goes to 1 on
  hover or keyboard focus. No hover state exists on touch, so the dimmed resting
  state is what makes copy/edit tappable there. Layout is unchanged (the footer
  already reserved `min-height`).

## Tests

- `agent-view.test.ts`: added a test asserting copy (assistant) and edit (user)
  render in their footers with no hover/interaction and are not `hidden` - guards
  against re-introducing a JS/`hidden` gate. The resting OPACITY itself is CSS,
  not computable from the webpack stylesheet under jsdom, so it is eyeball-verified
  in the served bundle (per `frontend-verify-needs-e2e-serve`); the built
  `dist/agent.js` was grepped to confirm it ships `opacity: 0.6` at rest and
  `opacity: 1` on hover/focus. 73 frontend tests green.
