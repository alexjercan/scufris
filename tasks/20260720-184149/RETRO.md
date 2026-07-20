# Retro: Settings UI - profile switcher + informative panels

- TASK: 20260720-184149
- BRANCH: feature/settings-panels
- REVIEW ROUNDS: 1 (APPROVE, NIT only)

## What went well

- Pure-additive on the T5 foundation: extending the `SettingsActions` seam with
  profile ops and adding an optional `SettingsExtras` bundle kept
  `renderSettings` pure, so every existing display test was unaffected and the
  new panels/switcher slotted in cleanly.
- Degraded states followed `stable-rows-with-dash-beats-conditional-sections`
  by construction (`infoPanel` always renders every row, `value ?? "-"`), so
  the "panels degrade to a dash" behavior was correct the first time and the
  reviewer confirmed no collapse/NaN (the context divide-by-zero guard held).
- Applied last task's lesson: staged explicit paths, no `git add -A`, symlink
  never leaked into the commit.

## What went wrong

- Nothing material. One small care point: both the profile "save as" and the
  MCP "add server" forms use `.settings__addserver`, so a test selecting by
  that class would grab the wrong one - selected by the input's aria-label
  instead.

## What to improve next time

- When two components share a CSS class, target test queries by a
  distinguishing attribute (aria-label), not the shared class.

## Action items

- No lessons ledger entry (all points are task-specific or already-banked
  habits). No follow-up code task.
- Manual DoD items moved to the umbrella GOAL.md Manual acceptance for the
  Finish checkpoint.
