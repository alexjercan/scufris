# Harden dashboard frontend: escape innerHTML + jsdom render test

- STATUS: OPEN
- PRIORITY: 10
- TAGS: feature,backlog,dashboard,ui,security

## Goal

Harden the dashboard frontend rendering: escape host-derived strings before
inserting them, and add a jsdom render smoke test.

## Notes

- From REVIEW.md / RETRO.md of tatr 20260719-154539 (both LOW, non-blocking).
- `web/src/main.ts` builds cards with `innerHTML` and interpolates values
  (hostname, os string, disk mountpoints). On a single-user local dashboard the
  risk is minimal, but a mountpoint containing an angle bracket would inject
  markup. Switch to `textContent`/element construction, or add an `escapeHtml`
  helper applied to every interpolated host-derived value.
- Add a jsdom-based smoke test that calls `renderCards`/`renderSummary` against a
  fixture `HostStats` and asserts the DOM (card count, values), closing the
  automated-render gap (no headless browser in the current setup).
- Keep `npm run ci` green.
