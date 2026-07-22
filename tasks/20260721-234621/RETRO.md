# Retro: U3 - unified per-agent settings PAGE component

- TASK: 20260721-234621
- BRANCH: feature/unified-settings-page (landed 47cfc5e)
- REVIEW ROUNDS: 1 (APPROVE, out-of-context; 3 findings adopted, 2 deferred/scope)

See TASK.md for what/why and REVIEW.md for the findings. Process notes only here.

## What went well

- The U3/U4 re-cut held: delivering the per-agent settings PAGE end-to-end via the
  SHARED detail shell's path-branch (no new webpack entry / backend route) meant a
  self-contained, reviewable slice with zero throwaway shim - the backend catch-all
  already served `/agents/<id>/settings`, so a regex branch in the entry was enough.
- Reused rather than reimplemented: exported `renderHealthCard` from settings-view
  and composed `agentFields`, so the settings page shares one health render + one
  field builder with the rest of the app (no drift).

## What went wrong

- I forgot to apply TWO lessons already in the ledger, and the gates/review caught
  them for me:
  - `interface-method-shorthand-trips-unbound-method`: I declared
    `AgentSettingsDeps.load()/save()` as method shorthand; eslint red on
    `unbound-method` when the deps were extracted. This is the exact lesson from
    20260721-180222 - I should have written them as function-typed properties up
    front.
  - `render-rewrite-orphans-its-css`: retiring the settings modal left the
    `.agent-modal*` CSS orphaned; the reviewer flagged it (R1). Removing a render
    surface must sweep its CSS in the same diff.

## What to improve next time

- Before writing a component whose area matches a ledger slug (frontend render
  rewrite; injected-deps interface), re-read those specific lessons and apply them
  DURING implementation, not after the linter/reviewer re-teaches them. The ledger
  only compounds if it is consulted forward, not just appended to.

## Action items

- [x] Adopted R1 (dead modal CSS), R2 (stale comment), R4 (never-run "not started").
- [x] Bumped `render-rewrite-orphans-its-css` to x2 (recurred here).
- [ ] R3 (per-agent page shows global/codex health) + R5 (writable hardcoded):
      deferred to U4/U5 with notes in REVIEW.md - no per-agent health source, and
      the read-only wiring is U4.
