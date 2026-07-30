# Add a Playwright and axe browser test harness

- STATUS: OPEN
- PRIORITY: 67
- TAGS: testing, v0.2.0, frontend, e2e, a11y
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT

## Story

As a developer, I want one hermetic browser-test command that boots Scufris,
drives a real browser, and shuts down cleanly, so that rendered behavior and
accessibility regressions are reproducible.

## Steps

- [ ] Add Playwright and axe dependencies, scripts, configuration, and browser
      installation/package wiring using the repository's Node/Nix toolchain.
- [ ] Build a fixture that allocates an unused port and temporary state
      directory, launches the packaged frontend plus FastAPI app, waits for
      readiness, records the PID, and always tears it down.
- [ ] Seed projects, agents, and deterministic mock scenarios through public
      APIs rather than reaching into in-memory stores.
- [ ] Add smoke coverage for every HTML route, static asset loading, initial API
      hydration, and nonblank rendered content.
- [ ] Fail on page errors, unexpected console errors, failed network requests,
      serious axe findings, and viewport overflow.
- [ ] Capture trace, screenshot, and server logs only on failure with paths
      suitable for local and CI diagnosis.
- [ ] Document the local command and prerequisites.

## Definition of Done

- The harness starts from a clean temporary state and leaves no helper process
  running (test: `browser-harness-lifecycle.spec.ts`).
- Every current page renders in Chromium with no unexpected browser errors
  (test: `route-smoke.spec.ts`).
- Axe and horizontal-overflow assertions run at desktop and mobile sizes
  (test: `accessibility-smoke.spec.ts`).
- The browser harness passes twice consecutively without leaked ports or state
  (cmd: `cd web && npm run test:e2e && npm run test:e2e`).

## Notes

- Epic: 20260729-102149.
- Depends on: 20260729-102151.
- V0.2.0 readiness role: every later orchestrator/project workflow must add a
  real-browser journey to this harness rather than relying on TypeScript unit
  rendering alone.
- Use recorded process handles for teardown. Never use `pkill -f`.
- Record browser packaging and lifecycle choices in `DECISION.md`.
