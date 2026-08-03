# Clear the round-1 MINOR findings from the diagnostics alignment

- PRIORITY: 40
- TAGS: chore, v0.2.0, telegram, frontend
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260729-102145

## Story

As a reader of the diagnostics contract, I want its pointers, wording and
comment style to match what the code actually does, so that the one section
that is supposed to be the source of the three-state vocabulary is not itself
stale.

## Steps

- [ ] R1.2 (RED FIRST) `tests/test_telegram_render.py:383` - extend
      `test_render_settings_summary_carries_the_capability_reading` with a
      quota whose `primary` is `None` and whose `secondary` is a real
      `RateWindow`, asserting the summary prints that window's percent and
      label. Confirmed red on the base branch in scratch: the summary prints
      `usage: nothing reported yet` for a secondary-only quota while
      `/settings usage` prints the window. Then fix
      `scufris/telegram/render.py:388` (`render_settings_summary`): read
      `usage.primary or usage.secondary` instead of `usage.primary`.
- [ ] R1.4 `scufris/telegram/render.py:325-341` (`render_usage`) - collapse the
      12-line `windows` comprehension and the redundant `usage is not None`
      guard at line 340 back to a direct guard. `usage` is provably not `None`
      past the early return, so the second guard is dead. Keep the primary
      -> secondary ordering and the `_fmt_window` labels: this body is what
      R1.2's new case is asserted to agree with.
- [ ] R1.1a `scufris/README.md:438` - the surface table cell names
      `web/src/agent-settings-view.ts (capabilityText)`. The page column is
      right; the symbol lives in `web/src/agent-settings-panels.ts:140`. Name
      the panels module for the symbol. (TASK.md says line 356 - stale; 438 is
      the live line.)
- [ ] R1.1b `scufris/telegram/text.py:58-59` - same wrong module in the
      three-state comment; point at `web/src/agent-settings-panels.ts`.
- [ ] R1.1c `web/src/agent-settings-view.ts:62,345` - both comments say "see
      `capabilityText`" with no module, from a file that does not define it.
      Qualify both with `agent-settings-panels.ts`.
- [ ] R1.3 `CHANGELOG.md` - add a bullet under `[Unreleased] ### Changed` for
      the three-state OPERATOR WORDING (`not reported by the <backend>
      backend` vs `nothing reported yet` vs a bare dash), naming the Telegram
      and web surfaces. The existing envelope bullet covers the API shape, not
      what the operator reads; this is a separate bullet under the same
      heading.
- [ ] R1.5 `web/src/agent-view.ts:151` - drop the `tasks/20260801-100419` task
      ID from the comment (AGENTS.md forbids task IDs in code). Keep the
      substance: this bar is the one surface that deliberately hides rather
      than printing the three-state sentence.
- [ ] R1.6 `web/src/agent-settings-panels.ts:45,140` - drop `export` from
      `capabilityPanel` and `capabilityText`; both are module-local
      (`capabilityPanel` at 79/87, `capabilityText` at 158/184). Do NOT touch
      `statusPanel` (line 114): the finding lists it, but
      `agent-settings-view.ts:32,215` imports and calls it - dropping that
      export breaks the build. The finding is 2/3 right.
- [ ] `resetsIn` dedupe - `web/src/chat-sidebar.ts:96` and
      `web/src/agent-settings-panels.ts:100` are byte-identical. Move one copy
      to `web/src/common.ts` (which both already import from) and import it in
      both. This also retires the third `export` R1.6 would otherwise leave.

## Definition of Done

- The `/settings` summary and `/settings usage` agree on a secondary-only
  quota (test: the R1.2 case in
  `test_render_settings_summary_carries_the_capability_reading`, red on the
  base branch - verified in scratch, prints `nothing reported yet` where `17%`
  is expected).
- No pointer sends a reader to the wrong module for `capabilityText`
  (cmd: `rg -n 'capabilityText' scufris/README.md | rg -v
  'agent-settings-panels'`, expected no match; 1 match on base.
  cmd: `rg -n 'agent-settings-view' scufris/telegram/text.py`, expected no
  match; 1 match on base.
  cmd: `rg -n 'capabilityText' web/src/agent-settings-view.ts | rg -v
  'agent-settings-panels'`, expected no match; 2 matches on base).
- The operator wording is in the changelog
  (cmd: `rg -n 'not reported by the' CHANGELOG.md`, expected a match; no match
  on base).
- No task ID survives in web source
  (cmd: `rg -n '20260801-100419' web/src --glob '!node_modules'`, expected no
  match; 1 match on base).
- Nothing is exported that no other module imports
  (cmd: `rg -n '^export function (capabilityPanel|capabilityText)'
  web/src/agent-settings-panels.ts`, expected no match; 2 matches on base).
- `resetsIn` exists once (cmd: `rg -n 'function resetsIn' web/src --glob
  '!node_modules'`, expected exactly one hit, in `common.ts`; 2 hits on base).
- No drift (cmd: `python -m pytest`; cmd: `nix flake check`; cmd: `cd web &&
  npm run ci`).

## Notes

The six open findings of `tasks/20260801-100419/REVIEW.md` Round 1. That round
APPROVEd; none blocks the branch, so they land as one cleanup instead of
reopening an approved diff.

- R1.1 `scufris/README.md:356`, `scufris/telegram/text.py:58`,
  `web/src/agent-settings-view.ts:62` still name `capabilityText` in
  `web/src/agent-settings-view.ts`; it lives in
  `web/src/agent-settings-panels.ts:140`.
- R1.2 `scufris/telegram/render.py:388` - `render_settings_summary` reads only
  `usage.primary`, so a quota with `primary=None` and a `secondary` window
  disagrees with `/settings usage`. Fall back to `secondary`, and extend
  `test_render_settings_summary_carries_the_capability_reading`.
- R1.3 no `CHANGELOG.md` bullet for the three-state operator wording.
- R1.4 `scufris/telegram/render.py:325` - the `windows` comprehension and the
  `usage is not None` guard at line 340 replace a 3-line guard with 12.
- R1.5 `web/src/agent-view.ts:151` cites a task ID in a code comment, which
  AGENTS.md:103 forbids.
- R1.6 `web/src/agent-settings-panels.ts:47,116,140` export three symbols no
  other module imports.

Also seeded by the retro, separable: `resetsIn` is duplicated verbatim in
`web/src/chat-sidebar.ts:96` and `web/src/agent-settings-panels.ts:116`.
Pre-existing; dedupe into `common.ts`.

Plan-time corrections - three of the six pointers had drifted, so the line
numbers in the findings above are not authority; the Steps carry the re-derived
ones:

- R1.1: `capabilityText` in `scufris/README.md` is at line 438, not 356.
- R1.6: `statusPanel` (line 114, not 116) IS imported and used; only
  `capabilityPanel` (45) and `capabilityText` (140) are unimported.
- `resetsIn` is at `agent-settings-panels.ts:100`, not 116.

The `resetsIn` dedupe is kept in scope rather than split off: ten lines, and it
touches the same two files R1.6 does, so splitting it would mean two commits
racing over `agent-settings-panels.ts`'s export list. It is independently
committable in principle - if the diff wants to stay pure-cleanup, it can be
dropped without touching anything else here.

R1.2 is the only item with a user-visible effect and the only one needing a
test; the rest are covered by the existing suites plus the greps above. Nothing
here moves an interface or a caller, so there is nothing load-bearing for a
DECISION.md.
