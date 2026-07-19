# Sparkline labels/tooltips + GPU VRAM bar placement

- STATUS: OPEN
- PRIORITY: 10
- TAGS: feature, dashboard, ui

## Goal

Two UI polish items on the just-landed sparklines (tatr 20260719-182915):

1. Each card's mini-graph should say what it means - a small visible label in the
   graph corner (btop-style) plus a hover tooltip (SVG `<title>`).
2. On the GPU card the VRAM fill bar currently sits ABOVE the "vram" text row,
   which reads as unrelated; move it to sit directly BELOW the vram numbers.

## Steps

- [ ] `stats-view.ts`: extend `sparkline(values, max?, sevClass?, title?)` to
      prepend an SVG `<title>` (hover tooltip) when `title` is given. Add a
      `labeledSpark(label, title, values, max?, sevClass?)` helper returning a
      `.spark-wrap` div = a `.spark__label` caption + the svg. Cards take the
      wrapper (HTMLElement).
- [ ] `renderCards`: build each card's graph via `labeledSpark` with a short
      corner label + a descriptive tooltip: CPU "cpu %" / "CPU utilization (%)",
      Load "load 1m" / "Load average (1 min)", GPU "gpu %" / "GPU utilization
      (%)", Memory "mem %" / "Memory used (%)", Disks "disk i/o" / "Disk I/O
      (read+write, bytes/s)", Network "net i/o" / "Network (up+down, bytes/s)".
- [ ] `gpuCard`: move `bar(gpu.mem_percent)` out of the top block and render it
      inside the detail rows immediately after the "vram" row, so the fill bar is
      directly under its numbers. Keep the util bar paired with the util value.
- [ ] `style.css`: `.spark-wrap` (position relative) + `.spark__label` (small,
      muted, absolute top-left over the graph, non-interactive). No card-height
      change.
- [ ] `stats-view.test.ts` (jsdom): a graph carries its `<title>` text and a
      `.spark__label`; the GPU card renders the VRAM bar after the vram row (bar
      is a following sibling of the vram row, not before it). `sparkline` unit
      tests still pass (title optional).
- [ ] LIVE serve smoke + `npm run ci` + `ruff`/`mypy`/`pytest` green.

## Definition of Done

- Every card graph shows a corner label and a hover tooltip explaining the
  metric; the GPU VRAM bar sits below the vram text. Card height unchanged; host
  strings escaped; jsdom tests + `npm run ci` + python checks green; serve-verified.

## Notes

- Follow-up to tatr 20260719-182915 (sparklines). User feedback: "add a label for
  each graph of what it means (tooltip on hover); vram fill bar should be below
  the vram text, not above."
