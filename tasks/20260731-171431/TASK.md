# Split the oversized frontend views under the size cap

- STATUS: OPEN
- PRIORITY: 75
- TAGS: refactor, v0.2.0, frontend, maintainability
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT
- PARENT: 20260731-171411
- DEPENDS ON: 20260731-171420

## Story

As a maintainer, I want the four oversized frontend modules split by view
concern, so that a page change does not load a thousand lines of unrelated
rendering and fetch code.

## Steps

- [ ] Characterize behavior with the existing `web/src/*.test.ts` suites before
      moving code.
- [ ] Split `web/src/agent-chat-view.ts` (1106) by stream handling, message
      rendering, and view wiring; reuse `chat-stream.ts`, `chat-format.ts`, and
      `markdown.ts` rather than duplicating them.
- [ ] Split `web/src/host-view.ts` (1022) by host section.
- [ ] Split `web/src/stats-view.ts` (870) by metric group and rendering.
- [ ] Split `web/src/common.ts` (834), keeping `apiFetch` and the auth bootstrap
      contract intact and unduplicated.
- [ ] Apply the epic comment policy to every file touched.
- [ ] Remove the corresponding allowlist entries from the size guard.

## Definition of Done

- No non-test file under `web/src/` exceeds 600 lines
  (cmd: `python scripts/check_file_size.py`).
- Frontend gate passes (cmd: `cd web && npm run ci`).
- Package build passes (cmd: `nix build .#scufris-web`).
- Authenticated calls still route through one `apiFetch`
  (cmd: `rg -n "apiFetch" web/src`).
- `web/README.md` conventions and file list match the new layout
  (cmd: `rg -n "view" web/README.md`).

## Notes

- Epic: 20260731-171411.
- Depends on: 20260731-171420.
- Test files move with their subject in the same change; the 900-line test cap
  applies to what remains.
