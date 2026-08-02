# Sparkline labels/tooltips + GPU VRAM bar placement

- PRIORITY: 10
- TAGS: feature, dashboard, ui
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Two UI polish items on the just-landed sparklines (tatr 20260719-182915):

1. Each card's mini-graph should say what it means - a small visible label in the
   graph corner (btop-style) plus a hover tooltip (SVG `<title>`).
2. On the GPU card the VRAM fill bar currently sits ABOVE the "vram" text row,
   which reads as unrelated; move it to sit directly BELOW the vram numbers.

## Steps

- [x] `stats-view.ts`: extend `sparkline(values, max?, sevClass?, title?)` to
      prepend an SVG `<title>` (hover tooltip) when `title` is given. Add a
      `labeledSpark(label, title, values, max?, sevClass?)` helper returning a
      `.spark-wrap` div = a `.spark__label` caption + the svg. Cards take the
      wrapper (HTMLElement).
- [x] `renderCards`: build each card's graph via `labeledSpark` with a short
      corner label + a descriptive tooltip: CPU "cpu %" / "CPU utilization (%)",
      Load "load 1m" / "Load average (1 min)", GPU "gpu %" / "GPU utilization
      (%)", Memory "mem %" / "Memory used (%)", Disks "disk i/o" / "Disk I/O
      (read+write, bytes/s)", Network "net i/o" / "Network (up+down, bytes/s)".
- [x] `gpuCard`: move `bar(gpu.mem_percent)` out of the top block and render it
      inside the detail rows immediately after the "vram" row, so the fill bar is
      directly under its numbers. Keep the util bar paired with the util value.
- [x] `style.css`: `.spark-wrap` (position relative) + `.spark__label` (small,
      muted, absolute top-left over the graph, non-interactive). No card-height
      change.
- [x] `stats-view.test.ts` (jsdom): a graph carries its `<title>` text and a
      `.spark__label`; the GPU card renders the VRAM bar after the vram row (bar
      is a following sibling of the vram row, not before it). `sparkline` unit
      tests still pass (title optional).
- [x] LIVE serve smoke + `npm run ci` + `ruff`/`mypy`/`pytest` green.

## Definition of Done

- Every card graph shows a corner label and a hover tooltip explaining the
  metric; the GPU VRAM bar sits below the vram text. Card height unchanged; host
  strings escaped; jsdom tests + `npm run ci` + python checks green; serve-verified.

## Implementation

- `sparkline(values, max?, sevClass?, title?)` prepends an SVG `<title>` when a
  title is given (native hover tooltip + a11y name), rendered even for empty
  data. New `labeledSpark(label, title, values, max?, sevClass?)` returns a
  `.spark-wrap` (position:relative) holding an absolute, `pointer-events:none`
  `.spark__label` corner caption + the svg; cards now take the wrapper.
- `renderCards` builds each graph via `labeledSpark` with a short caption + a
  descriptive tooltip (cpu %, load 1m, gpu %, mem %, disk i/o, net i/o).
- `gpuCard`: the VRAM `bar(mem_percent)` moved out of the top block into the
  detail rows right after the "vram" row, so the fill bar sits below its numbers;
  the util bar stays paired with the util value.
- `style.css`: `.spark-wrap` carries the graph's top margin (card height
  unchanged); `.spark__label` is a small muted mono chip with a semi-opaque bg.
- Tests: 3 new jsdom tests (sparkline `<title>`; per-card corner label + tooltip;
  GPU vram row's next sibling is the `.bar`). 30 jsdom total; `npm run ci` +
  `ruff`/`mypy` green; bundle carries `spark__label`/`spark-wrap`.

## Notes

- Follow-up to tatr 20260719-182915 (sparklines). User feedback: "add a label for
  each graph of what it means (tooltip on hover); vram fill bar should be below
  the vram text, not above."
