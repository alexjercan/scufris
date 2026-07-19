# Retro: Fill the Load card + fixed-size Disks

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- Asking the user which metrics they wanted in the Load card (and honoring that
  they skipped the load-vs-core bars) meant building exactly the card they asked
  for - not my first guess. The AskUserQuestion up front saved a review round.
- The stateful-collector + monotonic-rate pattern was reused a third time
  (`cpu_activity`), so context-switch/interrupt rates dropped in with no new
  machinery.
- Probing the real `disk_io` device names first (`loop0-7`, `nvme0n1`,
  `nvme0n1p1-3`) made the stable base-disk filter concrete: drop loop/ram/dm/sr
  by name and partitions via the strict-prefix rule. The jsdom test pins exactly
  that (nvme0n1 kept, partitions + loops dropped).
- Default-empty new fields kept every existing fixture/test working untouched -
  the compatibility seam paying off yet again.

## What went wrong / friction

- Nothing notable. One honest limit surfaced in review: the Network card has the
  same blink-resize behavior as Disks did, but the user only flagged Disks and
  showing ~12 mostly-idle nics would be noisier than helpful - left as a scoped
  follow-up rather than silently changing it.

## Lessons

- `stable-rows-with-dash-beats-conditional-sections`: a card that shows/hides
  subsections by "has data this poll" resizes and jars; render a STABLE row set
  (filtered once to the real entities) and show `-` for absent values instead. A
  `min-height` on the card damps the rest.

## Follow-ups

- Optional: apply the same stable treatment to the Network card if the interface
  blink annoys (show a base interface set, dash when idle).
- Header/footer fragments + polish (tatr 20260719-190549) is still open.
