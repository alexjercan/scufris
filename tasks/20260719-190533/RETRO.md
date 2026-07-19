# Retro: Rework stats cards (consolidate + route sensors)

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- The user's feedback was concrete enough to turn straight into a target card
  set in `/plan` (11 -> 6 cards), so the work was a focused rewrite rather than a
  design exploration - writing the target set into TASK.md first kept the rewrite
  honest against the ask.
- The earlier `common.ts` / side-effect-free `stats-view.ts` split made the
  rework painless: swap card builders, keep the module importable, and the 9
  jsdom tests assert the real new behavior (the "67" temp overlay on a square,
  swap text inside Memory, disk temp routed into Disks) - not just card counts.
- Routing sensors by where they belong (core temps on squares, nvme into Disks)
  killed the text-wall Temperatures card exactly as asked, and the escaping
  discipline carried over for free (both injection guards still green).

## What went wrong / friction

- The honest wrinkle: `coretemp` exposes fewer physical-core temps than logical
  CPUs (hyperthreading), so "a temp per load square" is an approximation - mapped
  by index proportion and documented in code + review, rather than pretending a
  1:1 core mapping exists. Worth surfacing to the user as an eyeball item.
- No headless browser here, so the final visual density (a number over a fill on
  up to 24 squares) is user-verified, not automated - the jsdom tests prove
  structure/values, not looks.

## Lessons

- `route-sensors-to-their-card-not-a-dump`: a flat "all sensors" card reads as a
  text wall; route each reading to the card it describes (core temps onto the CPU
  squares, drive temps into Disks) and consolidate related cards - fewer, denser,
  more intuitive. Reuse `card__subhead` to section a card (io / temp / swap).

## Follow-ups

- The header/footer fragments + polish task (tatr 20260719-190549) is the other
  half of the user's UI feedback.
- Sparkline history (tatr 20260719-182915) will add mini-graphs to these
  consolidated cards.
