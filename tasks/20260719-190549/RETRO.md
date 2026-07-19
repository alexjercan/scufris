# Retro: Header/footer as shared fragments + polish

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- Reading nova-protocol's `webpack-partials.js` first meant the plugin was a
  faithful, trimmed port rather than a reinvention - the beforeEmit hook +
  placeholder-replace pattern dropped straight into the existing multi-page
  webpack config with one `require` + one plugin instance.
- The multi-page split from the earlier task made this clean: each page template
  shrank to a placeholder + its page-specific middle, and the shared brand/nav/
  status now lives in one `_header.html`/`_footer.html`.
- The serve smoke verified the real thing that a build-only check misses: both
  built pages contain the injected header/footer and the placeholder is gone -
  proving single-source injection, not just "it compiled".

## What went wrong / friction

- A `set -e` in the smoke script aborted silently when a `grep -c` returned 0
  (exit 1) - the AGENTS.md shell rule again: a grep in a `$(...)` under `set -e`
  eats the run. Re-ran without `set -e`. Cheap, but a recurring foot-gun.

## Lessons

- `set-e-plus-grep-c-aborts-scripts`: under `set -e`, a `grep -c` (or any grep)
  that matches nothing exits non-zero and aborts the script, even inside
  `$(...)`. Use `grep -co ... || true`, drop `set -e` around greps, or test the
  count separately. (Restates the AGENTS.md "no pipe eats the exit code" rule for
  grep specifically.)

## Follow-ups

- The two UI-feedback tasks (card rework + this) are both done; the header/footer
  are now easy to extend with more nav items as pages are added.
- Remaining backlog: btop process view (182901), sparkline history (182915),
  agent chat-page spike (180528).
