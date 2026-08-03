# Clear the round-1 MINOR findings from the diagnostics alignment

- PRIORITY: 40
- TAGS: chore,v0.2.0,telegram,frontend
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -
- PARENT: 20260729-102145

## Story

As a reader of the diagnostics contract, I want its pointers, wording and
comment style to match what the code actually does, so that the one section
that is supposed to be the source of the three-state vocabulary is not itself
stale.

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
