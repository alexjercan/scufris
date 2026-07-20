# Agent sidebar: grouped, labeled sections (sessions / this session / account)

- STATUS: CLOSED
- PRIORITY: 30
- TAGS: feature, agent, ui, spike

## Goal

The sidebar reads as one undifferentiated column: the session list scrolls and
drags the context block + weekly meter with it, and nothing frames the three
distinct concerns. Group them into labeled, separately-behaving boxes (the user's
own example):

- **Sessions** - the chat history, in its own scroll area with a visible heading.
- **This session** - the context block (window fill %, tokens, turns/tools).
- **Account** - the weekly-usage meter.

So the history scroll never moves the stats, and each box says what it is. Also:
dedupe/relocate the cryptic head `ctx X · Y out` indicator (now redundant with
the context box), add a one-line explanation or tooltip per stat, and label the
usage "as of last turn" (codex only reports it mid-turn).

## Notes

- Spike: tasks/20260719-223054/SPIKE.md (P1) - this is the user's headline
  example.
- Consumes the existing `/api/agent/sessions|context|usage` endpoints; frontend +
  CSS only. Keep render side-effect-free for jsdom; escape session titles.
- Consider collapsible sections if vertical space is tight; keep the fixed-foot
  behavior so the stats stay visible.

## Implementation

- `index.html`: the session list is wrapped in a labeled
  `<section class="sidebar__section">` with a "Sessions" heading; the two stat
  boxes keep the pinned `.sidebar__foot` and gain `aria-label`s ("this session",
  "account"). The chat head's `#agent-usage` span is removed.
- `style.css`: `.sidebar__section` is a bordered, self-scrolling box that takes
  the sidebar's slack (`flex: 1`), so the history scroll never drags the stat
  boxes. `.sidebar__label` shares the `.usage-block__head` look; new
  `.usage-block__hint` styles the muted freshness footnote.
- `agent-view.ts`:
  - Dedupe: removed the head `ctx X · Y out` indicator and the now-dead
    `applyUsage`/`resetUsage` + `_cumulativeOutput`/`_lastContext` state. Every
    flow (turn/fork/switch/delete/new) already calls `refreshSidebar()`, which
    re-renders the context + account boxes from the API - the authoritative
    token source (and more accurate than the old client-side counter, which only
    summed turns done in the current tab).
  - `renderContext` head -> "this session" + a freshness hint "as of last turn".
    `renderUsage` head -> "account", the used row carries the window descriptor
    ("weekly"), + "as of last turn".
  - New `blockHead(text, tip)` / `blockHint(text)` helpers and a `tip` arg on
    `usageRow` set a native `title` tooltip per stat so the jargon is explained.

## Tests

- `agent-view.test.ts`: `renderContext` asserts the "this session" label, the
  freshness hint, and that the head + every stat row carry a non-empty `title`.
  `renderUsage` asserts the "account" label, "weekly" descriptor, and the hint.
  Removed the `applyUsage` suite (indicator deleted); dropped `#agent-usage` from
  the shared DOM fixture. 62 frontend tests green, webpack build clean; the built
  `dist/index.html` ships the section + labels and no longer ships `agent-usage`.
- The scroll-independence and tooltip hover are layout/interaction behaviors
  jsdom cannot measure - eyeball-verified in the served bundle (per
  `frontend-verify-needs-e2e-serve`).
