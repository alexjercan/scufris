# Agent chat: discoverability polish (tool chips, session tooltip, pill count, fork hint)

- STATUS: CLOSED
- PRIORITY: 20
- TAGS: feature,agent,ui
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Goal

Agent-page discoverability polish (grab-bag from the round-2 UX review):

- Make the per-turn tool-call chips prominent ("ran: host_stats" with clear
  styling, not a small low-contrast badge with a negative-margin hack).
- Add a full-title tooltip on truncated session rows.
- Show a count on the "new messages" pill (e.g. "3 new messages").
- Hint at fork/edit (branch a chat) in the onboarding empty state or the composer.

Do any subset in any order; split further at /plan if too broad for one cycle.

## Notes

- Spike: tasks/20260720-102348/SPIKE.md.
- Lowest priority of the round-2 tasks. Frontend-only. Keep render
  side-effect-free for jsdom; escape everything.

## Implementation (all four items)

- **Tool chips (`messageMeta` + CSS):** prepend a muted "ran" label before the
  tool chips, and restyle `.chat__chip` from a faint bordered pill to a filled
  accent badge; dropped the `.chat__meta` negative-margin hack for normal spacing.
- **Session tooltip (`renderSessions`):** set `open.title = session.title` (a
  property, so no attribute escaping) so a truncated row reveals its full title on
  hover.
- **Pill count (`renderLog`/`maybeScroll`/scroll listener/pill click):** track
  `_unreadCount` (+ `_prevMsgCount`); renderLog adds the message-count growth while
  the user is scrolled up, and the pill shows "N new message(s)" (or "jump to
  latest" when nothing is new). Reset to 0 whenever the user follows the bottom
  (stick), clicks the pill, or on new-chat/reset. New `refreshPill()` centralizes
  the label + visibility.
- **Fork hint (`renderWelcome` + CSS):** a `.chat__welcome-hint` line ("Tip: edit
  one of your messages to branch the conversation...") surfaces the otherwise
  undiscoverable fork feature.

## Tests / verification

- `agent-view.test.ts`: `messageMeta` has a `.chat__ran` label + chip (and none
  when usage-only); `renderSessions` sets the row `title`; the onboarding shows the
  fork hint; the pill counts messages that arrive while scrolled up ("2 new
  messages") and clears on jump. 82 frontend tests green; tsc clean.
- Visuals (chip prominence, hint, pill label) are eyeball-verified per
  `frontend-verify-needs-e2e-serve`.
