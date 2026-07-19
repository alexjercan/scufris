# Review: Harden dashboard frontend (escape innerHTML + jsdom tests)

## Round 1 - 20260719

Scope: `web/src/dashboard.ts` (new, from main.ts), `web/src/main.ts` (thin
entry), `web/src/dashboard.test.ts`, `web/package.json`, `web/vitest.config.ts`.

### Correctness

- The actual XSS surface is closed: `escapeHtml` is applied to every
  host-derived STRING that lands in `innerHTML` - disk mountpoints and the
  host/os value in the summary. These are element-content contexts, so escaping
  `< > &` (plus `"`) is complete; numeric values go through `toFixed` and can't
  inject; the chat panel already used `textContent`.
- The jsdom tests prove it, not just assert intent: a mountpoint of
  `"/<img src=x onerror=alert(1)>"` produces NO `<img>` element (literal text),
  and a `<script>` hostname produces no `<script>` - exactly the injection the
  prior review flagged. Plus escapeHtml unit coverage and a card-count/value
  check. 4 tests pass under `vitest run` (jsdom env).
- The main.ts -> dashboard.ts split is behavior-preserving: `start()` is the old
  `main()` verbatim; the entry is 3 lines. Serve smoke confirms the app still
  builds and serves (index + bundle 200, cards + chat present), so the refactor
  did not regress rendering.
- `npm run ci` now runs the tests (format:check + lint + test + build) and is
  green; `node_modules` stays gitignored, `package-lock.json` committed.

### Observations (non-blocking)

- LOW: `el(tag, class, html)` still assigns `innerHTML` in general, so safety
  relies on callers escaping host-derived values (now done). A stricter design
  would build text nodes, but the escaping + the injection tests cover the real
  cases without churn; fine for the single-user dashboard.
- NOTE: `escapeHtml` does not encode `'`; unnecessary here since no value is
  placed in a single-quoted attribute context (all are element content).

### Verdict

APPROVE. Meets the Definition of Done: host-derived strings are escaped, jsdom
tests prove a hostile mountpoint/hostname cannot inject DOM, `npm run ci` runs
them and is green, and the dashboard still builds and serves. LOW items are
appropriate to leave.
