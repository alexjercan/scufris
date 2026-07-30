# Harden dashboard frontend: escape innerHTML + jsdom render test

- STATUS: CLOSED
- PRIORITY: 10
- TAGS: feature, backlog, dashboard, ui, security
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Goal

Harden the dashboard frontend rendering: escape host-derived strings before
inserting them, and add a jsdom render smoke test.

## Steps

- [x] Add an `escapeHtml()` helper and apply it to every host-derived STRING
      interpolated into `innerHTML` (hostname, os_name, kernel, disk mountpoints).
      Numbers (percent/bytes via `toFixed`) are safe; the chat panel already uses
      `textContent`.
- [x] Refactor so render functions are importable without side effects: move the
      logic to `web/src/dashboard.ts` (exports `renderCards`, `renderSummary`,
      `escapeHtml`, `start`) and reduce `web/src/main.ts` to `import "./style.css";
      import { start } from "./dashboard"; void start();`.
- [x] Add `vitest` + `jsdom` devDeps, a jsdom `vitest.config`, a `test` script,
      and wire `test` into `npm run ci`.
- [x] `web/src/dashboard.test.ts`: `escapeHtml` escapes `< > & "`; `renderCards`
      with a malicious mountpoint (e.g. `"/<img src=x onerror=...>"`) injects NO
      element (escaped as text); `renderSummary` renders a `<script>` hostname as
      text; a basic card-count/value assertion.
- [x] `npm run ci` (format + lint + test + build) green; the app still builds and
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

## Implementation

- `escapeHtml()` added and applied to every host-derived STRING interpolated into
  innerHTML: disk mountpoints (`disksCard`) and the host/os value in
  `renderSummary`. Numbers (percent/bytes via `toFixed`) can't inject; the chat
  panel already used `textContent`.
- Refactored `web/src/main.ts` -> `web/src/dashboard.ts` (all logic, exported,
  no import-time side effects) + a 3-line `main.ts` entry (`import "./style.css";
  import { start } from "./dashboard"; void start();`), so render functions are
  importable by tests.
- Added `vitest` + `jsdom` devDeps, `vitest.config.ts` (jsdom env), a `test`
  script wired into `npm run ci` (format:check + lint + test + build).
- `web/src/dashboard.test.ts` (jsdom): `escapeHtml` escapes `< > & "`;
  `renderCards` with a hostile mountpoint injects NO `<img>` (stays literal
  text); `renderSummary` renders a `<script>` hostname as text; card-count/value
  assertions. 4 tests pass.
- Verified: `npm run ci` green (format + lint + 4 jsdom tests + build); the app
  still serves (index + bundle 200, cards + chat present).
