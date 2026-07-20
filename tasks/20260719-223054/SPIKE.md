# Spike: adversarial UX review of the agent page

- DATE: 20260719-223054
- STATUS: RECOMMENDED
- TAGS: spike, agent, ui

## Question

Now that the agent page has a chat, a session sidebar, a context block and a
weekly-usage meter, is it actually good to USE? Honest, adversarial: what works,
what is mediocre, what is bad, and would the target user (a technical homelab
operator self-hosting this) be happy? What are the concrete, prioritized
improvements - not vibes, specific changes.

## Context

The agent page (`web/src/index.html`, `agent-view.ts`, `style.css`) is a
two-pane shell: a left sidebar (`+ New chat`, a scrolling `#session-list`, and a
pinned foot with a context block + a weekly-usage meter) and a chat main pane
(head with model / a `ctx X · Y out` indicator / a `tools` toggle; a chat log; a
single-line input + send). The backend is `codex exec`, one blocking POST
`/api/chat` per turn (turn-based, NOT streamed). Assistant replies are rendered
with `msg.textContent = text` under `white-space: pre-wrap` - so NO markdown or
code formatting. Confirmed in code, not assumed.

## Options considered

This is a review, so the "options" are the candidate problems and whether each is
worth fixing. Grouped good / mediocre / bad, then converged into prioritized
tasks.

### What is going well (keep it)

- **Coherent identity.** The dark HUD theme, mono accents and consistent tokens
  look intentional and match the "scuffed Jarvis" idea. It does not look like a
  bootstrap default.
- **Information density that claude.ai/chatgpt deliberately hide.** Per-turn tool
  chips, token counts, live context-window fill %, and the weekly subscription
  quota are exactly what a homelab operator wants and cannot get in the official
  clients. This is the product's real edge.
- **Solid engineering hygiene.** Escaped host strings, responsive stack, no
  framework bloat, side-effect-free render, real session history on switch.

### What is mediocre

- **The head `ctx X · Y out` indicator is cryptic and now redundant** with the
  sidebar context block - two places show token/context numbers, neither
  explained. Jargon without a tooltip.
- **Empty/first-run state is bare.** A fresh chat is an empty log - no welcome,
  no example prompts, no hint that you can ask it to run tools. The sidebar list
  and blocks have no visible section labels ("Sessions", "This session",
  "Account"), only tiny uppercase micro-labels, so a newcomer does not know what
  the numbers are.
- **The chat head `Agent` title duplicates the nav** ("Agent" is already the
  active nav link) and eats vertical space.
- **Usage freshness is unlabeled.** The weekly meter is "as of the last turn"
  (codex only emits it mid-turn); after a pause it silently shows a stale figure.
- **No message timestamps**, so a returning user cannot tell what is old.

### What is bad (the daily-use pain)

- **No markdown or code rendering (worst gap).** This is an agent whose whole job
  is running CLI tools and returning code / command output. Replies come back as
  raw text: code blocks show literal backticks, lists show literal `- `, tables
  are mangled. `disk_usage`/`list_processes` output is at least monospace-ish
  under pre-wrap, but any prose-plus-code answer looks broken. For the primary
  interaction, this is the single biggest problem.
- **A dead "..." during long turns.** `codex exec` turns can take many seconds to
  minutes (it runs tools, reasons). The pending state is the literal string
  "...", no spinner, no elapsed time, no "running host_stats...", no cancel. The
  user cannot tell working-vs-hung. Combined with the blocking POST, a slow turn
  feels broken. (SSE was deferred in the earlier spike; the `--json` stream
  already emits per-item events we currently throw away until the turn ends.)
- **Single-line `<input>` for the prompt.** You instruct an agent with detail -
  multi-paragraph asks, pasted logs - and the composer is one line with no
  Shift+Enter newline, no autosize. Cramped and surprising.
- **No message affordances.** Cannot copy a reply or a code block, cannot retry a
  failed/hallucinated turn, cannot edit-and-resend. Baseline chat table stakes.
- **Scroll yank.** On reply the log force-scrolls to bottom (`log.scrollTop =
  scrollHeight`) even if the user scrolled up to read earlier context - it yanks
  them away. No "new messages" pill.
- **Sidebar grouping (the user's own example).** The context block and weekly
  meter are two `.usage-block`s pinned at the foot, but nothing frames the three
  distinct concerns - the SESSION LIST (history), THIS SESSION (context), and the
  ACCOUNT (weekly quota). They read as one undifferentiated column; a user must
  scroll the list past them. Grouping them into labeled, separately-scrolling
  boxes (history in its own scroll area; a fixed "this session" + "account"
  footer card) is exactly right and low-risk.
- **A11y gaps.** The chat log is not an `aria-live` region, so a screen reader
  never announces replies; focus is not moved to new content.

### Would a real user be happy? (honest)

Partly. A technical user would be **impressed by the dashboard-y richness** -
context %, quota, tool visibility are genuinely better than the official
clients - and would find it usable. But they would be **frustrated by the act of
chatting**: unformatted code replies, a dead "..." on long tool runs, a
one-line composer, and no copy button. Verdict: "great instrument panel, clunky
to actually talk to." Impressed, not delighted. The gap is entirely in the
core conversation loop, not the surrounding chrome - which is the good news,
because that is a bounded set of fixes.

### Do nothing

Viable only if this stays a personal toy the author alone uses and tolerates.
Given the stated goal (a real self-hosted assistant you chat with), the chat-loop
gaps are not deferrable - they are the product.

## Recommendation

Fix the conversation loop first, then the chrome. Five direction-level tasks,
prioritized:

1. **Markdown + code rendering (P0, tatr 20260719-223102)** - render assistant
   replies as sanitized markdown with fenced code blocks (mono, copy button).
   Biggest daily-use win. Keep user messages plain; keep escaping/sanitizing
   (untrusted model output into innerHTML needs a sanitizer, not raw set).
2. **Live turn progress / streaming (P0, tatr 20260719-223103)** - replace the
   dead "..." with real feedback: a working indicator, elapsed time, and live
   "running <tool>..." from the `codex exec --json` per-item events (harvest the
   stream we already run; SSE `/api/chat/stream` or chunked). Reduces the
   "is it hung?" anxiety that a turn-based agent creates.
3. **Multi-line composer (P1, tatr 20260719-223105)** - autosizing textarea,
   Enter=send / Shift+Enter=newline, clear disabled/sending state.
4. **Sidebar information architecture (P1, tatr 20260719-223106)** - the user's
   grouping: frame three labeled sections - Sessions (own scroll), This session
   (context), Account (weekly) - as distinct boxes so the list scroll never drags
   the stats; dedupe/relocate the cryptic head `ctx · out`; add a one-line
   explanation/tooltip per stat; label usage "as of last turn".
5. **Chat affordances + polish (P2, tatr 20260719-223111)** - copy on
   replies/code, message timestamps, a "scroll to bottom / new messages" pill
   that stops the yank, an onboarding empty state with example prompts, and
   `aria-live` on the log for screen readers.

P0s are the two that decide whether the core loop feels good; do them first. 3-5
are quality-of-life and can flow in any order after.

## Open questions

- **Streaming feasibility for task 2.** `codex exec --json` emits per-item events
  (`item.completed` incl. `mcp_tool_call`), but the earlier probe found it is
  TURN-level, not token-delta - so we can stream "tool started/finished" and a
  live elapsed timer, but not token-by-token text. Confirm during that task's
  /plan whether to (a) SSE the item events for tool-activity + a spinner, or
  (b) just a client-side elapsed timer + animated indicator with no backend
  change. (a) is richer; (b) is a one-file frontend fix. Pick per effort.
- **Markdown sanitizer choice (task 1).** A tiny vetted lib (e.g. a small
  markdown renderer + DOMPurify-style sanitize) vs a hand-rolled minimal
  renderer. The nix/webpack build tolerates a dep; weigh bundle size vs safety.
  Non-negotiable: model output is untrusted, so sanitize.
- Should session switching also show a small loading state (task 5 or its own)?

## Next steps

Direction-level tasks seeded (for `/plan` to break into steps):

- tatr 20260719-223102: Agent chat - render markdown + code blocks (P0)
- tatr 20260719-223103: Agent chat - live turn progress / streaming feedback (P0)
- tatr 20260719-223105: Agent chat - multi-line composer (P1)
- tatr 20260719-223106: Agent sidebar - grouped, labeled sections (P1)
- tatr 20260719-223111: Agent chat - affordances + polish (copy/timestamps/scroll/onboarding/a11y) (P2)

## Fix record

(Appended by each implementing task as it lands.)

- 20260719-223105 (multi-line composer, P1) - LANDED. Chat `<input>` -> autosizing
  `<textarea>`: Enter sends, Shift+Enter newlines, grows to a 200px cap then
  scrolls, disabled-while-sending preserved via a shared `submit()` that no-ops
  mid-turn. 62 frontend tests green. See tasks/20260719-223105/TASK.md.
- 20260719-223106 (sidebar sections, P1) - LANDED. Sidebar reframed into three
  labeled boxes (Sessions self-scrolls and takes the slack; This session +
  Account pinned) with per-stat tooltips and an "as of last turn" hint. The
  redundant head `ctx · out` indicator + its dead client counter were removed
  (the API-driven context box is authoritative). 62 frontend tests green. See
  tasks/20260719-223106/TASK.md.
- 20260719-223111 (affordances + polish, P2) - LANDED. Copy on replies,
  live+historical timestamps (new `TranscriptMessage.ts`), no-yank scroll with a
  "new messages" pill, an onboarding empty state with example prompts, and
  aria-live + focus a11y. 72 frontend + 123 pytest green. See
  tasks/20260719-223111/TASK.md.

All five seeded tasks are now LANDED - the "fix the conversation loop" arc
(markdown, streaming, composer, sidebar, affordances) is complete.
