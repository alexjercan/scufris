# F2: render agents as cards (Stats-style) + friendly backend labels + card->page nav

- STATUS: OPEN
- PRIORITY: 42
- TAGS: agents,frontend


## Goal

Render agents as CARDS (lift `card()`/`row()` from stats-view.ts): a `.cards`
grid where each card shows name, friendly backend label, state badge, project,
and live turns/tokens. Clicking a card navigates to `/agents/<id>`. Wire the
friendly backend labels (from B1) into the list + create picker.

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (recommendation F2; reuse map).
- Depends on: 20260721-112429 (B1, labels), 20260721-112433 (F1, nav target).
