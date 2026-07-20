# Agent chat: discoverability polish (tool chips, session tooltip, pill count, fork hint)

- STATUS: OPEN
- PRIORITY: 20
- TAGS: feature,agent,ui

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
