# Review: Fill the Load card + fixed-size Disks

## Round 1 - 20260719

Scope: `scufris/metrics.py` (process_count + cpu_activity), `tests/test_metrics.py`,
`web/src/common.ts` (types), `web/src/stats-view.ts` (load + disks),
`web/src/stats-view.test.ts`, `web/src/style.css` (min-height).

### Correctness

- Load card now carries the confirmed set: the 1/5/15 headline plus tasks
  (process count), ctx-switches/s, interrupts/s, and uptime. Live-verified:
  process_count 511, ctx ~20k/s, interrupts ~6k/s. The card is no longer empty.
- `cpu_activity` is computed as a proper rate: `psutil.cpu_stats()` deltas over a
  persisted monotonic timestamp; first sample is 0 (no prior), later samples
  divide by real elapsed time (tested across two samples). Consistent with the
  net/disk rate pattern.
- Disks stability: the `io` section now renders a STABLE base-disk set and shows
  `-` when idle instead of hiding rows/sections, so the card stops resizing as IO
  blinks. The base-disk filter is sound - drops `loop*/ram*/dm-*/sr*` and
  partitions via the strict-prefix rule (a jsdom test proves `nvme0n1` stays
  while `nvme0n1p1` and `loop0` are dropped). `.card` min-height reduces
  collapse/jump.
- Back-compat kept: both new `HostStats` fields default-empty, so the existing
  conftest fixture and app tests are untouched. Host-derived names still escaped;
  render module still side-effect-free. python + `npm run ci` (11 jsdom) green.

### Observations (non-blocking)

- LOW: the Network card still uses the active-only + slice filter, so it can
  still resize as interfaces blink - the user only called out Disks, and showing
  all ~12 nics (mostly idle veth/docker) would be noisier than helpful. The
  `.card` min-height dampens it; a stable base-interface treatment is a candidate
  follow-up if it annoys.
- LOW: a one-time layout shift at startup remains - the first poll has no
  disk_io/rates yet (no prior counters), so the io section appears on the second
  poll. Not the per-poll blinking the user reported; acceptable.
- NOTE: min-height is a fixed 150px; very tall cards (Disks with many rows) still
  grow, which is expected - the goal was to stop the jump/collapse, not force
  identical heights.

### Verdict

APPROVE. Meets the Definition of Done: the Load card is full (load + tasks +
activity + uptime), the Disks card shows a stable base-disk set with a dash when
idle and no longer resizes on IO blink, cards have a min-height; live-verified,
names escaped, checks green. LOW items are scoped follow-ups.
