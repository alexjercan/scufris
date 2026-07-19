# Agent chat: message affordances and polish (copy, timestamps, scroll, onboarding, a11y)

- STATUS: OPEN
- PRIORITY: 20
- TAGS: feature, agent, ui, spike

## Goal

A grab-bag of table-stakes chat quality-of-life, currently all missing:

- **Copy** button on assistant replies (and on code blocks once markdown lands).
- **Timestamps** on messages so a returning user can place them.
- **Scroll behavior**: stop the yank - only auto-scroll when the user is already
  at the bottom; otherwise show a "new messages / scroll to bottom" pill (today
  `log.scrollTop = scrollHeight` fires on every reply and rips the user away from
  scrolled-up history).
- **Onboarding empty state**: a fresh chat shows a short welcome + a few example
  prompts (e.g. "what's using my CPU?", "list open tatr tasks") instead of a
  blank log.
- **A11y**: make the chat log an `aria-live="polite"` region so screen readers
  announce replies; move focus sensibly on new chat.

## Notes

- Spike: tasks/20260719-223054/SPIKE.md (P2).
- Independent of the others; can be split further at /plan if it is too broad for
  one cycle. Copy-on-code-block pairs naturally with the markdown task
  (20260719-223102).
- Keep render side-effect-free for jsdom; escape everything.
