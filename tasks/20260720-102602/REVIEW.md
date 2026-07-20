# Review: chat discoverability polish

- VERDICT: APPROVE
- ROUND: 1

## Summary

Four grab-bag items from the round-2 UX review: a "ran" label + filled tool chips
(from a faint badge), a full-title tooltip on truncated session rows, a live count
on the "new messages" pill, and a fork/edit tip in the onboarding. 82 frontend
tests green; tsc clean.

## What is good

- Tool chips now read as output, not clutter: a muted "ran" label + filled accent
  chips, and the `.chat__meta` negative-margin hack is gone (proper spacing). This
  is the item the reviewer weighted most - tool execution is the point of the agent.
- The pill count is correct across the tricky flows, not just the happy path:
  `_unreadCount` grows only when `_messages` actually grows while `!_stickToBottom`,
  and resets on every follow-the-bottom path (submit/fork/switch set stick=true;
  reset/newChat zero it; the scroll listener and pill-click clear it). Only the
  final `onDone` render grows `_messages`, so a streaming reply counts once, and a
  user's own turn (sent while stuck to bottom) never counts. Pinned with a test
  that scrolls up, adds two messages, asserts "2 new messages", and clears on jump.
- Session tooltip uses `open.title = ...` (a property), sidestepping attribute
  escaping entirely - the right way to set untrusted text as a tooltip.
- The fork hint surfaces a genuinely undiscoverable feature (edit-to-branch) in the
  one place a new user looks (the empty state).

## Findings

- MINOR (accepted) - `index.html`'s static `#chat-jump` text is still "new
  messages"; `refreshPill()` overwrites it on first show (and the pill starts
  hidden), so it is never seen. Left as-is rather than touch the markup.
- MINOR (accepted, pre-existing corner) - deleting the *current* session while
  scrolled up does not force stick=true, so a stale count could linger for one
  frame; the next scroll event corrects it, and the scenario is vanishingly rare.

## Verdict

APPROVE. Four small, self-contained improvements, each tested at the level jsdom
allows; the one with real logic (the pill count) is correct across the mutation
paths and pinned. Visuals are eyeball per `frontend-verify-needs-e2e-serve`.
