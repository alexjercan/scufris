# Review: Richer host metrics (GPU, sensors, freq, net/disk rates)

## Round 1 - 20260719

Scope: `scufris/metrics.py` (models + collector), `tests/test_metrics.py`,
`web/src/common.ts` (types), `web/src/stats-view.ts` (cards),
`web/src/stats-view.test.ts`, `web/src/style.css`.

### Correctness

- Proven live on this host: `/api/stats` returned the real RTX 3060 Ti (26%,
  370/8192 MB, 41 C, 15 W), coretemp/nvme/acpitz temps, 24 per-core freqs, and
  12 net + 12 disk rate rows on the primed poll. The stats page serves and
  renders the new cards. Real data, end to end.
- GPU via the CLI (injectable `gpu_runner`, `shutil.which` + subprocess +
  timeout) matches the spike's NixOS-robust choice; `parse_gpus` tolerates
  missing output and `[N/A]` fields (tested). Empty list when absent - degrades
  cleanly on a GPU-less host.
- Rates are computed correctly: the collector persists previous per-NIC/per-disk
  counters + a MONOTONIC timestamp, so the first sample carries no rates and
  later samples divide the delta by real elapsed time (tested across two
  samples). Sorted by throughput; negatives clamped to 0 (counter resets).
- Backward compatible: the new `HostStats` fields default to empty, so the
  existing `conftest` fixture and app tests keep working untouched.
- Escaping held: every host-derived name (GPU, NIC, disk, sensor chip/label) goes
  through `escapeHtml`; a jsdom test proves a hostile GPU name injects no element.
- ruff + mypy + pytest and `npm run ci` (6 jsdom tests) are green.

### Observations (non-blocking)

- MINOR: `nvidia-smi` is spawned on EVERY `/api/stats` poll (~2s). It runs in
  FastAPI's threadpool (sync route), so it does not block the loop, but a short
  cache (e.g. 1s) would cut subprocess churn if the poll interval drops. Fine at
  the current cadence.
- MINOR: the GPU card stacks two bars (util then VRAM) without an inline label
  between them; the value shows util% and the `vram` row sits below, so it reads,
  but a small label per bar would be clearer. Cosmetic.
- MINOR: the Temperatures card can get tall (coretemp exposes ~14 per-core rows);
  acceptable, could be collapsed later. `row()`/`el()` keep the innerHTML pattern
  (callers escape) - consistent with the existing code, covered by the injection
  test.

### Verdict

- VERDICT: APPROVE

Meets the Definition of Done: `/api/stats` carries GPU, temps, fans,
per-core freq and net/disk rates from real host data (degrading to empty when
absent, rates correct across polls); the stats page shows the new cards with
escaped names; checks green; live-verified on this host. MINOR items are polish /
future-tuning.
