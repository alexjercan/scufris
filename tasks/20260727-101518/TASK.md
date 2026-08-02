# Match sub-agent Tools section to orchestrator tool cards

- PRIORITY: 1
- TAGS: web, ui, settings
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Problem

On a sub-agent's settings page, the `Tools (N)` section renders as a flat
key/value list (`settings__row` rows: name on the left, description on the
right). On the orchestrator's settings page, the `Tools (N)` section renders as
a grid of cards (`tool-grid` of `tool-card`s: cyan name, uppercase server badge,
description, and an `args:` line). The two look inconsistent even though both
show `AgentTool[]` data. The sub-agent Tools section should look like the
orchestrator's tool cards.

## Done Means

1. The sub-agent settings page renders its `Tools (N)` section as a
   `tool-grid` of `tool-card`s, visually identical to the orchestrator's read-only
   tool cards (name in cyan, server badge, description, args line), while staying
   read-only (no toggle checkbox, no "Try it" runner). (test: `web` vitest suite)
2. The empty state ("none (this backend exposes no scufris tools)") still shows
   for an agent whose backend exposes no tools. (test: existing empty-state test)
3. The orchestrator page is unchanged (it still uses the writable console). No
   stray checkbox/runner appears on the sub-agent Tools cards. (test: existing
   "read-only ... no toggle/checkbox" test)
4. `npm run ci` (lint + typecheck + vitest) is green. (cmd: `nix develop --command bash -c 'cd web && npm ci && npm run ci'`)

## Approach

The orchestrator's card is built by the private `toolCard(tool)` in
`web/src/settings-view.ts:63` (used inside `renderToolControls` ->
`tool-grid`). The sub-agent's `agentToolsPanel` in
`web/src/agent-settings-view.ts:137` hand-rolls `settings__row`s instead.

Reuse, do not duplicate (avoid drift):

- Export `toolCard` from `settings-view.ts`.
- In `agentToolsPanel`, keep the `settings__card` section wrapper + `Tools (N)`
  title, but replace the `settings__row` loop with a `tool-grid` div containing
  one plain `toolCard(t)` per tool. `toolCard` alone has no toggle/runner, so the
  panel stays read-only by construction.
- Keep the empty-state path (the `panel(...)` "none" note) as is.

`AgentTool` already carries `server` and `args` (common.ts:207), and the
sub-agent's `agentTools` come from `/api/agents/{id}/tools` as `AgentTool[]`, so
no new data is needed.

## Steps

- [x] Export `toolCard` from `web/src/settings-view.ts`.
- [x] Refactor `agentToolsPanel` in `web/src/agent-settings-view.ts` to render a
      `tool-grid` of `toolCard(t)` inside the `settings__card` section.
- [x] Update/extend `agent-settings-view.test.ts` to assert the tool-card
      structure (`.tool-grid` / `.tool-card` / `.tool-card__name` present, still
      no checkbox) and keep the empty-state assertion.
- [x] Run `nix develop --command bash -c 'cd web && npm ci && npm run ci'` green.
