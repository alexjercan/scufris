# Review: sparkline labels/tooltips + GPU VRAM bar placement

- DATE: 20260719
- VERDICT: APPROVE (1 round)

## Scope reviewed

`web/src/stats-view.ts` (sparkline `<title>` + `labeledSpark` wrapper + card
wiring + gpuCard bar move), `web/src/stats-view.test.ts` (3 new tests),
`web/src/style.css` (`.spark-wrap` / `.spark__label`). No backend change.

## Correctness

- `sparkline` gained an optional 4th `title` arg that prepends an SVG `<title>`
  (native hover tooltip + a11y name), rendered even for the empty-data case so a
  still-filling graph already hints. All prior positional calls and the four
  existing `sparkline` unit tests are unaffected (title defaults to "").
- `labeledSpark` wraps the svg in a `.spark-wrap` (position:relative) with an
  absolutely-positioned, `pointer-events:none` `.spark__label` caption. The
  wrapper carries the top margin the bare `.spark` used to, so card height is
  unchanged and the caption cannot intercept clicks.
- GPU VRAM bar: moved out of the top block into the detail rows immediately after
  the "vram" row, so the fill bar reads as belonging to its numbers. The util bar
  stays paired with the util value/graph. Verified by a test asserting the vram
  row's `nextElementSibling` is a `.bar`.
- Labels are static strings (no host data), so nothing new to escape; the
  existing hostile-mountpoint / hostile-GPU-name escape tests still pass.
- Full suite green: `npm run ci` (format + lint + 30 jsdom tests + build) and
  `ruff`/`ruff format`/`mypy` in the dev shell; pytest unaffected (no Python
  changed). Built `stats.js` carries `spark__label`/`spark-wrap`.

## Nits (non-blocking)

- The corner caption slightly overlaps the top of the graph line at high values;
  the semi-opaque background chip keeps it legible. Acceptable; user eyeball.

## Verdict

APPROVE. Meets the DoD: every card graph shows a corner label + hover tooltip
naming its metric, and the GPU VRAM bar now sits below the vram text. Card height
unchanged, host strings escaped, all checks green, bundle-verified.
