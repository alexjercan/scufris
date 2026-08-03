# Fix responsive layout and accessibility audit findings

- PRIORITY: 0
- TAGS: bug,backlog,frontend,ui,a11y
- KIND: TASK
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Story

As a keyboard, screen-reader, or mobile user, I want Scufris controls and
content to remain reachable and correctly framed, so that the current visual
design works beyond a wide desktop mouse session.

## Steps

- [ ] Add failing browser regressions for the audited navigation overflow,
      home-page autofocus scroll, clipped mobile composer, nested interactive
      agent card, low-contrast usage hint, and unfocusable process scroller.
- [ ] Make navigation wrap, scroll intentionally, or collapse behind a familiar
      control without losing the current page indication.
- [ ] Prevent initial composer focus from moving a mobile viewport while
      preserving sensible desktop and post-send focus behavior.
- [ ] Replace the button-role card nesting with valid link/button semantics and
      complete keyboard activation.
- [ ] Meet WCAG AA contrast and make scrollable process content keyboard
      focusable with a visible focus state.
- [ ] Audit all routes at representative desktop, tablet, and mobile widths for
      text clipping, overlap, touch targets, and horizontal scroll.

## Definition of Done

- Every audited page has `scrollWidth <= clientWidth` at 390px
  (test: `mobile-layout-regressions.spec.ts`).
- Initial mobile navigation to chat stays at the top of the page
  (test: `chat_does_not_autoscroll_on_mobile_load`).
- Axe reports no serious A/AA violations on current routes
  (test: `accessibility-smoke.spec.ts`).
- Agent cards and the process scroller are fully keyboard operable
  (test: `keyboard-navigation.spec.ts`).
- Desktop density and terminal character remain intact (manual: user check).

## Notes

- Epic: 20260729-102157.
- Depends on: 20260729-102152.
- V0.2.0 readiness role: repair the known mobile, focus, semantic, contrast,
  and overflow baseline before the actor-aware conversation and project
  workspace add denser cross-page controls.
- Known code locations include `web/src/agent-chat-view.ts`,
  `web/src/agents-view.ts`, `web/src/processes-view.ts`, and
  `web/src/style.css`.
