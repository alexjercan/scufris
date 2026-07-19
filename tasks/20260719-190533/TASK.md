# Rework stats cards: consolidate + route sensors into their cards

- STATUS: OPEN
- PRIORITY: 12
- TAGS: feature,backlog,dashboard,ui

## Goal

Rework the stats page cards to be fewer, denser and more intuitive: dissolve the
text-heavy Temperatures card by routing each sensor to the card it belongs to,
and merge related cards. Keep the current visual style (the user likes it) - this
is a layout/consolidation pass, not a restyle.

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
