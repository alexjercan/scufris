# Retro: Richer host metrics (GPU, sensors, freq, net/disk rates)

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- The spike had already resolved the hard calls (nvidia-smi over NVML, psutil
  sensors, current-first), so this was a clean build against a decided design.
  Capturing a REAL nvidia-smi CSV line first meant the parser and its test were
  written against actual output, not a guess.
- Reusing the stateful `Collector` seam paid off again: it already primed cpu%,
  so adding persisted net/disk counters + a monotonic clock for rates fit the
  same pattern, and the injectable `gpu_runner` let the GPU path be tested with a
  canned sample (no GPU needed in CI).
- Defaulting the new `HostStats` fields to empty kept every existing fixture and
  the app tests working with zero churn - the compatibility seam paid for itself.
- The live check surfaced real numbers across the board (RTX 3060 Ti, coretemp,
  24-core freq, interface rates), which is the only way to know rates actually
  populate on the second poll.

## What went wrong / friction

- mypy caught a real bug: reusing `cur`/`prev` loop variables across the net and
  disk loops conflicted the psutil namedtuple types (`snetio` vs `sdiskio`).
  Renamed the disk-loop vars. Worth remembering: distinct names per loop when the
  element types differ.

## Lessons

- `distinct-loop-vars-for-different-types`: don't reuse a loop variable name
  across two loops whose elements are different (nominal) types - mypy binds one
  type to the name and the second loop's attribute access fails. Name them apart.
- `capture-real-cli-output-for-parser-tests`: when parsing a CLI's output, run it
  once and pin a real captured line as the test fixture, so the parser is written
  against reality (nvidia-smi's exact CSV, including `[N/A]`).

## Follow-ups

- The btop-style process view (tatr 20260719-182901) and the deferred sparkline
  history (tatr 20260719-182915) build on this.
- Optional polish from REVIEW.md: a short nvidia-smi cache if the poll interval
  drops; per-bar labels on the GPU card; a collapsible Temperatures card.
