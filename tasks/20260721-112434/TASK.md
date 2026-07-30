# F2: render agents as cards (Stats-style) + friendly backend labels + card->page nav

- STATUS: CLOSED
- PRIORITY: 42
- TAGS: agents,frontend
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED


## Goal

Render agents as CARDS (lift `card()`/`row()` from stats-view.ts): a `.cards`
grid where each card shows name, friendly backend label, state badge, project,
and live turns/tokens. Clicking a card navigates to `/agents/<id>`. Wire the
friendly backend labels (from B1) into the list + create picker.

## Steps

- [x] Add `BACKEND_LABELS` + `backendLabel()` to `web/src/common.ts` (Codex/Claude/Mock).
- [x] Rewrite `web/src/agents-view.ts`: render agents as a `.cards` grid of
      clickable `.agents__card`s (name + state badge + backend label + project +
      mode + live turns/tokens or "not started"); card click -> `open(id)` nav.
- [x] Simplify `AgentActions` to `{create, remove, open, reload}` and drop the
      in-page detail panel / select / status / SSE machinery (now on the F1
      detail page); keep a per-card delete button (stops propagation).
- [x] Use `backendLabel()` in the create-form backend picker.
- [x] `startAgents`: poll `/api/agents` + each agent's status, feed the cards a
      `statuses` map, focus-guard the re-render; `open` -> `location.assign`.
- [x] Add `.agents__card` / `.agents__card-del` CSS (clickable card, hover ring).
- [x] Rewrite `web/src/agents-view.test.ts` for the card API (12 tests).

## Definition of Done

- Agents render as a `.cards` grid of cards; each card shows name, friendly
  backend label, state badge, project and turns/tokens
  (test: `renders agents as cards`, `shows live turns/tokens from a running status`).
- The friendly backend label is shown, not the raw id, on cards and in the
  create picker (test: `shows the friendly backend label, not the raw id`).
- Clicking a card navigates to `/agents/<id>` via the injected `open` action;
  the delete button does not navigate
  (test: `opens the agent page when the card is clicked`,
  `deletes an agent (with confirm) without navigating`).
- The full web gate passes (cmd: `npm run ci` in web/).

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (recommendation F2; reuse map).
- Depends on: 20260721-112429 (B1, labels), 20260721-112433 (F1, nav target).
- Close-out: cards show turns/tokens from a per-agent status poll (the list
  endpoint carries only `state`), so `renderAgents` takes a `statuses` map and
  `startAgents` fans out cheap status fetches. Live SSE/events stay on the F1
  detail page - the list page is glanceable, not a control surface. Description
  is intentionally NOT on the card (it lives on the detail page); the card keeps
  to the task's named fields. Whole-card click via a `role=button`/`tabindex`
  section with an Enter/Space keydown; the delete button `stopPropagation`s so
  it never also navigates.
- Review: 1 round, out-of-context APPROVE. Addressed 2 MINOR + 2 NIT: keyboard
  delete no longer navigates (target guard on the card keydown), removed dead
  CSS orphaned by the rewrite, made the turns test assertion load-bearing, and
  trimmed the XSS test to the name-only case it exercises.
