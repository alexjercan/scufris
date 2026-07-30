# Projects UI: Projects page (list + create + project detail with tatr tasks)

- STATUS: CLOSED
- PRIORITY: 25
- TAGS: feature,projects,ui
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

As the operator, I want a Projects page in scufris to see my projects, create
one, and open it to see its details and tatr tasks - the visible home of the
projects-orchestrator concept.

## Steps

- [x] Add a new page `/projects/`: `web/src/projects.html`, a `projects` entry
      in `web/webpack.config.js` (+ one `HtmlWebpackPlugin` with `chunks:
      ["projects"]`, mirroring the settings page), and a `web/src/projects.ts`
      thin entry that calls `startProjects` (no import-time side effects).
- [x] Add `Project` and `ProjectTask` interfaces to `web/src/common.ts`
      (mirror the backend models).
- [x] `web/src/projects-view.ts`: a PURE `renderProjects(root, projects,
      selected, tasks, actions)` (jsdom-testable, no fetch) that renders the
      project list, a create form (name/cwd/language/description), and - when a
      project is selected - a detail panel with its metadata and its tatr tasks.
      Escape every host/user string (`escapeHtml`). `startProjects` does the
      fetch orchestration (list projects; on select, fetch
      `/api/projects/{id}/tasks`) and wires create/delete via `sendJson`.
- [x] Link the page into the app nav (wherever the other pages
      (stats/settings/agent) are linked from) so it is reachable.
- [x] jsdom tests (vitest): renders the project list + create form; selecting a
      project shows its detail + tasks; a hostile project name/description is
      escaped (no injection); create submits the form values.
- [x] Verify end to end: `npm run ci`, then serve the built bundle through the
      backend and confirm `/projects/` lists/creates a project and shows its
      tatr tasks (`frontend-verify-needs-e2e-serve`).

## Definition of Done

- `renderProjects` shows the list + create form, and a selected project's
  detail + tatr tasks (test: `projects_page_renders_list_and_detail`).
- A hostile project name/description injects no markup
  (test: `projects_page_escapes_hostile_strings`).
- `npm run ci` passes in `web/` (cmd: `cd web && npm run ci`).
- End-to-end: `/projects/` lists/creates a project and shows its tasks on the
  running backend (manual: load `/projects/`, create a project pointing at a
  dir with tatr tasks, see them).

## Notes

- Depends on: 20260720-210644 (CRUD API) and 20260720-210645 (tasks endpoint).
- Reuse the settings page patterns: pure render + injected actions seam
  (`SettingsActions`-style), `sendJson` for mutations, single authoritative
  render (reload from server after a mutation).
- Lessons: `webpack-multipage-htmlplugin-per-page` (one entry + plugin per
  page; FastAPI `StaticFiles(html=True)` serves `/projects/` with no backend
  change); `type-change-fails-strict-tsc` (run full `npm run ci`, grep every
  new-type literal); `escape-only-host-strings-in-element-content`; symlink
  `web/node_modules` into the worktree, and NEVER `git add -A`.
- Entry points: `web/src/settings-view.ts` (pattern), `web/webpack.config.js`
  (entries ~13, plugins ~45), `web/src/common.ts`.

## Close-out

- New `/projects/` page: `projects.html` + `projects.ts` entry + a webpack
  entry/HtmlWebpackPlugin/historyApiFallback rewrite (the multipage lesson);
  `Project`/`ProjectTask` types in common.ts; a Projects nav link in
  `_header.html` (+ `initNav` marks it active).
- `projects-view.ts` follows the settings-page pattern: a PURE `renderProjects`
  (list + create form + a selected project's detail with metadata and its tatr
  tasks) driven by an injected `ProjectActions` seam; `startProjects` does the
  fetch orchestration (list; on select, fetch that project's tasks) and wires
  create/delete via `sendJson`, reloading from the server after each mutation.
- Tasks are lazy: selecting a project renders its detail immediately with a
  "loading tasks..." state, then fetches `/api/projects/{id}/tasks` and
  re-renders. A failed tasks fetch degrades to an empty list, not a blank page.
- Every host/user string is escapeHtml'd (project name/desc/cwd/language, task
  title/tags); pinned by two hostile-input jsdom tests.
- Verified per `frontend-verify-needs-e2e-serve`: served the built bundle and
  confirmed `/projects/` 200, create -> id=demo, list, the demo project's REAL
  tatr task (p30) shown, and the nav link present. Isolated + cleaned the state
  dir and demo project.
- Staged explicit paths (no git add -A); symlink removed before landing.
- 125 frontend tests (6 files) + webpack build green.
