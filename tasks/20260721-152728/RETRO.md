# Retro: F5 agent detail chat-first reshape

- TASK: 20260721-152728
- BRANCH: feature/agent-detail-chat-first
- REVIEW ROUNDS: 1 (out-of-context APPROVE, 2 NITs addressed)

## What went well

- The out-of-context reuse-map (Explore) before coding was worth it: it settled
  the layout classes to reuse (.agent-shell / .sidebar / .usage-block / .bar),
  which landing stat boxes lift (context) vs need data we lack (usage/account),
  and confirmed the AgentRunStatus fields available - so the build was mechanical
  and the account box was deferred with a documented reason instead of shipping
  the wrong (codex-landing) account data on a claude agent.
- Splitting the old monolithic renderAgentDetail into two pure functions
  (renderSidebar, renderSettingsModal) kept everything jsdom-testable and made
  the poll-vs-persistence story trivial: the sidebar is polled, the settings form
  lives in a separate modal root, so a mid-edit cannot be wiped. That reused the
  F4 lesson (persistent-widget-needs-its-own-root) cleanly.
- Reapplied two prior ledger lessons without re-paying for them: the
  flex-defeats-the-hidden-attribute guard on `.agent-modal[hidden]`, and the
  separate-root pattern for the chat.

## What went wrong

- R1.1 (NIT): I attached the modal backdrop-close listener via
  `root.addEventListener` INSIDE the re-rendered `renderSettingsModal`, so every
  open stacked another listener on the persistent modal root. Benign (idempotent
  onClose) but a leak. Root cause: treating a re-rendered element like a
  fresh one - addEventListener accumulates across renders. Fixed by using the
  `onclick` PROPERTY (assignment replaces, never stacks).
- R1.2 (NIT): the test rewrite dropped a still-valid "blank name is a no-op"
  assertion. Re-added. Root cause: rewriting a test file wholesale risks losing
  coverage of guards that survived the refactor - diff the old test list.

## What to improve next time

- On an element that is RE-RENDERED in place, register handlers via the
  `on<event>` property (overwrites) rather than `addEventListener` (stacks),
  unless you also remove the prior listener. Ledgered.
- When rewriting a test file for a refactor, list the OLD test names first and
  check each still-valid behavior is re-covered, so a surviving guard does not
  silently lose its test.

## Action items

- [x] Both NITs addressed in round 1.
- Next: F6 (per-backend model dropdown), then the exec-drop + docs task.
