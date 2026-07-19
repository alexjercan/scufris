# Review: Rework stats cards (consolidate + route sensors)

## Round 1 - 20260719

Scope: `web/src/stats-view.ts` (card rework), `web/src/stats-view.test.ts`,
`web/src/style.css` (core-temp overlay + subhead).

### Correctness

- Matches the feedback point-for-point: core temps overlay the CPU load squares
  (number + color, load still the fill); frequency folds into the CPU card; swap
  folds into Memory; one Disks card carries usage + IO + temp; one Network card
  carries live rates + since-boot totals; the standalone Temperatures card is
  gone; Load average stays separate (>= 2 CPU cards kept). Card count drops
  ~11 -> 6 (+ per GPU).
- The 9 jsdom tests assert the actual new behavior, not just card counts: the
  consolidated set (5 cards for the base fixture, no Swap/Temperatures/frequency
  titles), swap text inside Memory, the core-temp overlay VALUE ("67") on a
  square with folded "GHz" + "package", and disk IO ("nvme0n1") + temp ("nvme
  Composite") routed into Disks.
- Escaping preserved: every host-derived string (mount, gpu, nic, disk device,
  sensor label) still goes through `escapeHtml`; both injection guards pass.
- Render module stays side-effect-free (tests import it); ruff/mypy/pytest and
  `npm run ci` green; serve smoke on this host (/, /stats/, bundles, /api/stats
  all 200).

### Observations (non-blocking)

- LOW / documented: physical `coretemp` readings (fewer than logical CPUs) are
  mapped across the load squares by INDEX PROPORTION - both hyperthreads of a
  physical core show that core's temp. An approximation, called out in a comment
  and the task; falls back to load-only when no "Core*" readings exist.
- LOW: disk-temp routing keys off a chip name containing `nvme`/`drivetemp`/
  `disk`; a differently-named drive-temp chip wouldn't be routed into Disks (would
  simply not appear). Reasonable heuristic for the common case.
- NOTE: visual density (a number over a fill on up to 24 squares) is exactly what
  the user asked for; the text-shadow keeps it legible. Final look is the user's
  to eyeball - no headless render here.

### Verdict

APPROVE. Meets the Definition of Done: the consolidated card set is in place with
no standalone Temperatures/Swap/Disk-IO/Network-interfaces/CPU-frequency cards,
sensors are routed to their cards (core temps on the squares, disk temp in Disks),
Memory shows swap, Network is unified; style preserved, names escaped, checks
green. The LOW items are documented approximations appropriate to the data.
