# Harden dashboard frontend: escape innerHTML + jsdom render test

- STATUS: OPEN
- PRIORITY: 10
- TAGS: feature,backlog,dashboard,ui,security

## Goal

Harden the dashboard frontend rendering: escape host-derived strings before
inserting them, and add a jsdom render smoke test.

## Steps

- [ ] Add an `escapeHtml()` helper and apply it to every host-derived STRING
      interpolated into `innerHTML` (hostname, os_name, kernel, disk mountpoints).
      Numbers (percent/bytes via `toFixed`) are safe; the chat panel already uses
      `textContent`.
- [ ] Refactor so render functions are importable without side effects: move the
      logic to `web/src/dashboard.ts` (exports `renderCards`, `renderSummary`,
      `escapeHtml`, `start`) and reduce `web/src/main.ts` to `import "./style.css";
      import { start } from "./dashboard"; void start();`.
- [ ] Add `vitest` + `jsdom` devDeps, a jsdom `vitest.config`, a `test` script,
      and wire `test` into `npm run ci`.
- [ ] `web/src/dashboard.test.ts`: `escapeHtml` escapes `< > & "`; `renderCards`
      with a malicious mountpoint (e.g. `"/<img src=x onerror=...>"`) injects NO
      element (escaped as text); `renderSummary` renders a `<script>` hostname as
      text; a basic card-count/value assertion.
- [ ] `npm run ci` (format + lint + test + build) green; the app still builds and
      serves (quick smoke).

## Definition of Done

- Host-derived strings are HTML-escaped before insertion; a jsdom test proves a
  hostile mountpoint/hostname cannot inject DOM.
- `npm run ci` runs the jsdom tests and is green; the dashboard still builds and
  serves.

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
