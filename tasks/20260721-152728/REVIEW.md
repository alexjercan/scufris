# Review: F5 agent detail chat-first reshape

- TASK: 20260721-152728
- BRANCH: feature/agent-detail-chat-first

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Full web gate ran green in the worktree (prettier + eslint + vitest 163 + webpack
build). Verified: startAgentDetail targets #agent-sidebar/#agent-settings-modal
and bails if absent; the poll re-renders ONLY the sidebar so a mid-edit modal
form survives; chat is untouched in its own root; stat-box percent math + empty
states correct; the `.agent-modal[hidden]` flex-defeats-hidden guard is present;
XSS handled; close-out matches the code. Two NITs, both addressed below.

- [ ] R1.1 (NIT) agent-detail-view.ts:235-237 - the backdrop-click listener is
  added via `root.addEventListener` inside the re-rendered `renderSettingsModal`,
  so each open stacks another (benign, but a leak).
  - Response: fixed - use `root.onclick = ...` (property assignment overwrites,
    so re-render replaces rather than stacks the handler).
- [ ] R1.2 (NIT) agent-detail-view.test.ts - the rewrite dropped the
  "does not save when the name is blanked" test though the guard still exists.
  - Response: fixed - re-added a blank-name no-op test to the renderSettingsModal
    suite.
