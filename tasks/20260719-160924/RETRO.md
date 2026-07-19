# Retro: Harden dashboard frontend (escape innerHTML + jsdom tests)

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- The fix matched the threat precisely: only host-derived STRINGS in
  element-content contexts needed escaping (mountpoints, host/os), so
  `escapeHtml` on `< > & "` is complete; numbers and the already-`textContent`
  chat path needed nothing. No over-engineering into full DOM construction.
- Splitting `main.ts` into a side-effect-free `dashboard.ts` + a 3-line entry
  made the render functions importable, which is what unblocked real jsdom
  tests. The tests assert the actual injection is prevented (no `<img>`/`<script>`
  element), not just that a helper exists.
- `vitest` + `jsdom` dropped into the existing TS/webpack project cleanly and
  wired into `npm run ci`, closing the "no automated render coverage" gap the
  original dashboard retro flagged.

## What went wrong / friction

- Nothing notable. One deliberate call: kept the general `el(..., html)`
  innerHTML helper rather than rewriting every card to text nodes - the escaping
  + injection tests cover the real surface, and a full rewrite would have been
  churn for a single-user local dashboard.

## Lessons

- `escape-only-host-strings-in-element-content`: for interpolating into
  innerHTML, escape only the untrusted STRINGS, and only for the context they
  land in (element content needs `< > &`; attributes also need quotes). Numbers
  via `toFixed` are safe. Prove it with a jsdom test that a hostile value creates
  no element, not just that a helper is called.
- `side-effect-free-module-for-jsdom-tests`: to unit-test frontend render logic,
  keep it in a module with NO import-time side effects (no auto-`start()`, no CSS
  import) and a thin entry file that wires it up - otherwise importing the module
  under vitest kicks off `fetch`/timers.

## Follow-ups

- None. This was the last open backlog task; the three product pillars
  (monitoring, agent chat, agent tool-running) are all functional and
  live-verified.
