# Review: message affordances visible at rest

- VERDICT: APPROVE
- ROUND: 1

## Summary

`.chat__copy, .chat__edit` resting opacity 0 -> 0.6, and the brighten rule now
includes `:hover` on the buttons plus the existing `:focus`. Copy (reply) and edit
(user) are now visible at rest - so they work on touch (no hover) and are never
invisible with a mouse - and brighten to the accent on hover/focus. Built bundle
confirmed to ship `opacity: 0.6` at rest / `1` on hover-focus. 73 frontend tests
green.

## What is good

- Minimal, surgical: one CSS rule pair, no markup or JS change, layout unchanged
  (the footer already reserved height). Low risk of regressing the other affordance
  work.
- Covers the whole ask: touch (dimmed resting state is tappable), keyboard (focus
  brightens), mouse (visible + hover feedback). Edit is included alongside copy for
  consistency, per the task.
- The test pins the right jsdom-level invariant (buttons rendered, not `hidden`, no
  hover needed) and the retro/task are honest that the opacity itself is CSS and
  eyeball/grep-verified, per `frontend-verify-needs-e2e-serve`.

## Findings

- MINOR (not blocking) - `.chat__copy:hover` / `.chat__edit:hover` are technically
  redundant with `.chat__foot:hover` (the button lives inside the footer, so
  hovering it already matches the footer rule). Kept for explicitness; harmless.
- MINOR (accepted) - every message now shows a dimmed "copy"/"edit" label, which is
  slightly busier than the previous invisible-until-hover. That is the intended
  trade for discoverability (the user's complaint); 0.6 keeps it quiet.

## Verdict

APPROVE. It does exactly what the task asks with the smallest possible change and
is honest about what jsdom can and cannot verify. The two findings are cosmetic.
