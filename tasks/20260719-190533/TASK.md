# Rework stats cards: consolidate + route sensors into their cards

- STATUS: OPEN
- PRIORITY: 12
- TAGS: feature,backlog,dashboard,ui

## Goal

Rework the stats page cards to be fewer, denser and more intuitive: dissolve the
text-heavy Temperatures card by routing each sensor to the card it belongs to,
and merge related cards. Keep the current visual style (the user likes it) - this
is a layout/consolidation pass, not a restyle.

## Plan / target card set (2026-07-19)

Consolidate from ~11 cards to ~6 (+ one per GPU). Final set:

- **CPU** (keeps its own card): headline cpu% + bar; per-core squares now show
  the core TEMPERATURE as a number, colored by temp, with load still the fill
  height; fold in average frequency (a row) and the package/system temp. Removes
  the standalone "CPU frequency" card.
- **Load average**: stays its own card (satisfies "keep >= 2 CPU cards").
- **GPU** (per GPU): unchanged.
- **Memory**: merge Swap into it (mem headline + bar, then swap used/total + %).
  Removes the standalone "Swap" card.
- **Disks**: ONE card with per-mount usage (bars) + per-device IO rates + disk
  temperature (nvme). Removes the standalone "Disk IO" card.
- **Network**: merge the since-boot totals + per-interface active rates into one
  card. Removes the standalone "Network interfaces" card.
- Removes the standalone **Temperatures** card entirely (its readings now live in
  CPU/Disks; any stray chip like acpitz shown compactly in CPU).

## Steps

- [ ] CPU card: overlay per-core temp (number + color) on the load squares -
      map logical cores to the physical `coretemp` "Core*" readings by index
      proportion (note the approximation); add avg-freq and package/system temp
      rows. Remove the standalone frequency card.
- [ ] Memory card: fold Swap in (used/total + percent rows). Remove the Swap card.
- [ ] Disks card: one card = per-mount usage + bars, per-device IO rates
      (`disk_io`), and disk temps (nvme group). Remove the Disk-IO card.
- [ ] Network card: merge since-boot totals + active per-interface rates. Remove
      the Network-interfaces card.
- [ ] Remove the Temperatures card; route sensors as above; keep `escapeHtml` on
      every host-derived name/label. Update `renderCards` ordering + any CSS
      (core temp overlay).
- [ ] Update the jsdom tests to the new layout (card set, core-temp overlay,
      swap-in-memory, disk temp/IO in Disks, escaping still holds).
- [ ] LIVE serve smoke: the stats page shows the consolidated cards with real
      data (core temps on squares, swap in Memory, IO+temp in Disks, unified
      Network). `ruff`/`mypy`/`pytest` + `npm run ci` green.

## Definition of Done

- The stats page renders the consolidated set above; there are NO standalone
  Temperatures, Swap, Disk-IO, Network-interfaces or CPU-frequency cards.
- Core squares show temperature (number + color) with load as fill; Memory shows
  swap; Disks shows usage + IO + temp; Network is one card. Style preserved.
- Serve-verified on this host; host-derived names escaped; `npm run ci` (jsdom)
  + python checks green.

## Feedback captured (user, 2026-07-19)

- The `Temperatures` card is too much text. Figure out what each sensor is and
  move it into its relevant card:
  - CPU core temps -> show them ON the per-core "load squares" as numbers +
    colors (today the squares encode cpu% by fill height + a tooltip; combine
    cpu% and temperature per core there).
  - Disk temp (nvme Composite) -> into the Disks card.
  - Motherboard/ACPI (acpitz, asus) + package temp -> fold into the CPU card (or
    a small system line); decide during design.
- Consolidate cards:
  - ONE Disks card with everything disk-related: usage + IO rates + temperature
    (merge today's separate `Disks` and `Disk IO`).
  - Memory card should INCLUDE swap (merge `Memory` + `Swap`).
  - Consolidate the Network things into one card (today's `Network` totals +
    `Network interfaces` rates).
  - There are 3 CPU-ish cards now (CPU util, CPU frequency, Load average); reduce
    redundancy but KEEP AT LEAST 2 SEPARATE (user's constraint) - e.g. fold
    frequency into the CPU card, keep Load average on its own.
- Net effect: fewer, well-packed, intuitive cards; the style stays.

## Notes

- Current stats UI: `web/src/stats-view.ts` (cards) + `web/src/style.css`; the
  richer metrics (GPU/sensors/freq/net/disk) landed in tatr 20260719-182846 and
  are all available in `HostStats`.
- Keep host-derived names escaped (existing lesson). Keep the render module
  side-effect-free for the jsdom tests; update those tests to the new layout.
- Related: the btop process view (tatr 20260719-182901) is a separate card/area;
  sparkline history (tatr 20260719-182915) will later add mini-graphs to these
  consolidated cards - design them with room for that.
- Layout judgement is involved; a quick mockup/design pass during /plan is worth
  it before rewiring the cards.
