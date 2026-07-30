# Notes: the dashboard host approval queue

What shipped, and the two things the build corrected. The decision this page
renders rather than makes is `tasks/20260729-125040/DECISION.md` sections 3 and 6.

## What shipped

`web/src/host-view.ts` (pure `renderHost` + a thin `startHost`), `host.ts`,
`host.html`, a `host` webpack entry and page, the dev-server rewrite, a nav link,
the `.host__*` block in `style.css`, and the API types in `common.ts`.

Three things the page does that are worth naming:

- **It renders the requirement, it does not invent one.** `record.confirmation`
  comes from the backend and says what approving costs: the risk phrase, whether an
  undo exists, and the acknowledgement token a one-way action needs. A one-way
  action therefore has NO ordinary approve control on its card - the only button
  that can approve it is the one that sends the token, disabled until the operator
  types it - so the proportionate friction is structural rather than a checkbox
  someone could route around, and the service refuses a tokenless approve anyway.
- **No HTML sink.** Every string on this page is attacker-influenceable (a systemd
  unit is named by a FILE; a preview quotes store paths, journal lines and command
  output). `text()` and `line()` are the only ways text is set, both via
  `textContent`, and `el()` is never called with its html argument. The escaping
  test compares the SAME page rendered with a hostile string and with a harmless
  one and asserts an identical element count - so it fails if any sink appears
  (verified by temporarily switching `text()` to `innerHTML`: one test red).
- **The edges are rendered, not hidden.** Expired and drifted proposals show why
  they can no longer be decided INSTEAD of controls the server would refuse; a
  failed multi-step apply says how far it got and warns against reading it as
  "nothing happened"; a 409 from the other surface appears as news; and a queue
  with nothing in it says so rather than looking broken.

## The two corrections the build forced

**1. "Not configured" cannot be read off the queue endpoint.** The first version
treated a 503 from `/api/host/actions` as "this box has no helper". Serving the app
for real and curling it showed that endpoint answers `200 []` when the helper is
absent (the app's own registry is simply empty) - only `/api/host/audit` answers
503. So the page would have shown an empty queue and an empty log, reading as
"nothing has been asked of it yet" when the truth is that nothing CAN be. The
signal is now the audit read, the server's own sentence (which names the env vars
and the NixOS module) is shown, and `startHost` got its only two tests because the
mistake lived in the orchestration layer that convention leaves untested.

This is the ledger's `frontend-verify-needs-e2e-serve` earning its place again: the
webpack build, 245 vitest tests and the whole python suite were green with this
bug in.

**2. The step's premise about a backend route was wrong.** The plan said to add "the
FastAPI page route and its `historyApiFallback` rule". `StaticFiles(html=True)`
already serves `/host/` from `dist/host/index.html`, and that is also what makes
the page protected by the deny-by-default middleware - a route would have been dead
code. Only the multi-page webpack wiring and the dev-server rewrite were needed.
The step was rewritten to say what shipped.

## Difficulties

- **The API-seam guard caught the SSE stream**, as designed: `new EventSource(` is
  forbidden outside an allowlist, because it cannot carry the CSRF header. Added
  with the same reasoning as `chat-stream.ts`, plus the point that matters here -
  every DECISION still goes through `apiFetch`, and the queue poll behind the
  stream means a dropped stream costs progress output, never the result.
- **The strict tsc build caught a closure-narrowing error vitest did not**
  (`type-change-fails-strict-tsc-not-vitest`): `document.getElementById` returns
  `HTMLElement | null`, and the early-return guard does not narrow it inside the
  nested `render`/`refresh` functions. Bound to a non-null const instead.
- **The first `escaping` assertion was wrong, not the code.** `expect(innerHTML).not.toContain("<img")`
  failed because the risk badge sets `title` as a DOM PROPERTY - never parsed as
  markup - so the serialised attribute legitimately holds the raw string. The
  structural element-count comparison is both correct and stronger.

## Self-reflected feedback

- **Serve it before believing the shapes.** Both real defects here were about what
  the SERVER actually answers, not about render logic, and neither the type
  definitions nor the test doubles could have caught them - the doubles were built
  from the same wrong assumption as the code. Curling the two endpoints took a
  minute and would have been worth doing before writing `readQueue`.
- **When a plan step names a mechanism, check the mechanism exists.** "Add the
  FastAPI page route" survived from planning into the step list unchallenged; two
  minutes reading how `/stats/` is served would have removed it at plan time.
- **A structural assertion beats a string search for "did this data become
  markup".** Counting elements against a clean render says exactly what matters and
  does not fight the DOM's own serialisation.
