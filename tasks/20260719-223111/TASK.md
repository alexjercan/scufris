# Agent chat: message affordances and polish (copy, timestamps, scroll, onboarding, a11y)

- STATUS: CLOSED
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

## Implementation

All five items landed in one cycle (cohesive chat-log polish, mostly one file):

- **Copy on replies**: a `copyButton(getText, cls)` helper (clipboard-guarded,
  flips to "copied" for 1.2s) is placed in each assistant message's footer and
  copies the raw markdown. Code-block copy already shipped with the markdown
  task; this adds whole-reply copy.
- **Timestamps**: `LogEntry` gains `ts?: number` (epoch ms). Live turns stamp
  `Date.now()`; historical turns get a real time from a new
  `TranscriptMessage.ts` (backend `read_transcript` reads each rollout event's
  top-level `timestamp`). `formatTimestamp` renders "HH:MM" same-day / "Mon D,
  HH:MM" older into a `<time datetime title>` in the footer.
- **No-yank scroll**: `_stickToBottom` (maintained by a log scroll listener via
  `isNearBottom`) gates auto-scroll; `maybeScroll` replaces the unconditional
  `scrollTop = scrollHeight` everywhere. When the user has scrolled up, new
  content reveals a "new messages" pill (`#chat-jump`) instead of yanking; a
  user action (send/fork/switch) re-pins. A `_rendering` guard stops the
  rebuild's own scroll events from mis-reading the follow state.
- **Onboarding**: a fresh log (`_agentEnabled && _messages empty`) renders a
  `.chat__welcome` with example-prompt chips; clicking one fills the composer
  (never auto-sends).
- **A11y**: `#chat-log` is `role="log" aria-live="polite" aria-relevant="additions"`;
  the composer is focused on load and on new chat.

## Tests

- Frontend: `formatTimestamp` (same-day/older/empty), `isNearBottom` (pinned /
  slop / scrolled-up via defined scroll metrics), copy-button copies raw text,
  timestamp element present/absent, onboarding welcome + chip-fills-composer,
  composer focused on load, and the pill reveal-on-scroll-up + jump-hides-it.
  71 frontend tests green.
- Backend: `read_transcript` carries an event's top-level timestamp onto `ts`
  (None when absent). 123 pytest green. The built `dist/index.html` ships the
  aria-live log, role=log, and the jump pill.
- Layout-dependent behavior (autoscroll feel, pill placement, hover reveals) is
  eyeball-verified in the served bundle per `frontend-verify-needs-e2e-serve`;
  the wiring is unit-tested by defining scroll metrics and dispatching events.
