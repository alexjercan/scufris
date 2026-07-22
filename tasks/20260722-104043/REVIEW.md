# Review: project detail page /projects/<id>

Out-of-context review of the branch (feature/project-detail-page) against the DoD.
Reviewer read the actual diff + the agent-detail reference + the static mount + the
CSS, and ran both suites.

- VERDICT: APPROVE (one dead-CSS NIT adopted)

## Findings

- NIT: `.projects__item--active .projects__name` (style.css) was dead once the
  inline selection (`projects__item--active`) was removed. Adopted: deleted it, and
  since the registered project name is now an `<a>`, added `text-decoration: none`
  + a cyan `:hover` (reusing the old active accent) so it does not read as a
  default link.
- NIT (no action): the delete button's `aria-label` + `window.confirm` use the raw
  project name - safe (attribute/confirm are not HTML-parsed); the visible `<h1>`
  name IS escaped. Matches the pre-existing agent-detail behavior.

## Verified clean (reviewer)

- Routing/shadowing: the dynamic `/projects/{id}` + `/{id}/{rest:path}` are
  registered before the static mount (mounted LAST); a single-segment param does
  not steal bare `/projects/` (static list), and `/api/projects/{id}` is a distinct
  earlier prefix. All bundles are root-absolute (`publicPath: "/"`), so nothing
  lives under `/projects/<id>/...` for the catch-all to intercept. The backend test
  proves the shell is served, `/projects/` returns the static list, and
  `/api/projects/{id}` 404s (not the shell) - a faithful mirror of agent-detail.
- DoD: the registered name is a real `<a href="/projects/<id>">` (id
  encodeURIComponent'd); all three sections render; each agent links to its page;
  delete DELETEs then `window.location.assign("/projects/")`.
- Consolidation: the inline panel + `select`/`remove` actions are fully removed
  with no dangling refs; `dispatch` + `Project` still used; `startProjects` still
  compiles; the obsolete fixtures + detail tests were removed.
- Frontend: `projectIdFromPath` returns null for `/projects/` and `/agents/x`;
  the agents filter is `project_id === id`; `maybe()` best-effort means a failing
  panel cannot blank the page; re-render on async arrival via the `render()`
  closure. Escaping present on every render path (name, cwd, description, agent
  state, task title/tags); agent name via `textContent` (safe).
- Tests non-vacuous: XSS tests assert no `img`/`b` element AND the literal text;
  delete tests assert both confirm gates; link-href tests assert exact hrefs; the
  backend test would fail if the route shadowed the list or the API.
- ASCII-only: no new non-ASCII; the back link uses `<-`.
