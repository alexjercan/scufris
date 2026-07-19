# Review: btop-style sparklines on the stats cards

- DATE: 20260719
- VERDICT: APPROVE (1 round)

## Scope reviewed

`web/src/stats-view.ts` (history buffer + `sparkline` + card wiring),
`web/src/stats-view.test.ts` (6 new tests), `web/src/style.css` (`.spark*`).
No backend change.

## Correctness

- Client-side ring buffer (`_history`, `HISTORY_LEN=60`, `push`) is bounded and
  keyed per series (cpu/load/mem/disk/net/gpu:<i>); `push` caps with a `while`
  shift so it can never exceed the cap. `_resetStatsHistory` clears it and is
  called in the test `beforeEach`, so module state does not leak across cases
  (the persistent-ui-state-needs-a-test-reset-hook lesson).
- `sparkline` is pure and empty-safe: no data -> an `<svg>` with no polyline (the
  first-poll degenerate case renders nothing jarring, then fills). Percent series
  clamp y to 0-100 via `max=100`; rate/load autoscale to the window max floored
  at 1, so an idle disk/net or a sub-1.0 load sits as a low flat line instead of
  amplifying noise - matches btop's autoscale behaviour.
- Host strings are untouched by this change (sparklines are pure numbers), and
  the existing hostile-mountpoint / hostile-GPU-name escape tests still pass.
- Full suite green: `npm run ci` (format + lint + 27 jsdom tests + build) and
  `ruff`/`ruff format`/`mypy`/`pytest` in the dev shell. Serve smoke: `/stats/`
  200, `/api/stats` real, and the built `stats.js` contains `spark__line`/
  `spark__area`/`_resetStatsHistory`.

## Nits (non-blocking)

- The GPU card puts the util sparkline between the util bar and the VRAM bar.
  It reads fine (the graph is the util history, sitting under its bar), but a
  future tweak could group both bars then the graph. Aesthetic only; deferred to
  the user's eyeball pass.
- No `aria`/`<title>` on the SVG. The numeric values are already shown as text
  beside every graph, so the sparkline is decorative; acceptable.

## Verdict

APPROVE. The feature meets the Definition of Done: every card (CPU, Load, GPU,
Memory, Disks, Network) carries a btop-style mini-graph that fills across polls,
percent cards fixed to 0-100 and rate/load cards autoscaled, card height stable
across polls, host strings still escaped, all checks green and serve-verified.
Visual polish is the user's call.
