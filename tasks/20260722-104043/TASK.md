# project detail page /projects/<id> - tasks, registered agents, metadata

- STATUS: IN_PROGRESS
- PRIORITY: 40
- TAGS: projects, frontend, backend

## Goal

Clicking a project on `/projects` should open a `/projects/<id>` detail page
showing useful project info: the project's tatr tasks, the agents registered to
it, and its metadata (cwd, language, description, ...), rather than only listing
projects on one page with no drill-in.

## Why

User feedback (2026-07-22): "when I click on a project it opens it in a
`/projects/<id>` page where I can see more details (all the tatr tasks, the
agents registered, metadata etc more useful things about the project)". Gives
each project a real home, mirroring the per-agent page.

## Notes / scope to pin

- New route `/projects/<id>` (SPA fallback + an entry/mount, like the per-agent
  detail shell). Needs backend endpoints for a project's tasks + registered
  agents if not already present.
- The user also noted the "register" button sits far from the name and is easy to
  missclick, but said to DISREGARD it (all projects should be registered anyway) -
  do not spend effort on that.
- Probably a /spike to decide what the detail page shows and which endpoints feed
  it (tatr integration for tasks).

## Spike findings (mapped 2026-07-22, folded in - no separate SPIKE task)

The backend is almost entirely already there; this is mostly a frontend page +
two SPA shell routes, mirroring the per-agent detail shell.

- ALREADY EXISTS (reuse, no change): `GET /api/projects/{id}` -> Project (404),
  `GET /api/projects/{id}/tasks` -> list[ProjectTask] (parses `tatr -r <cwd> ls`
  via `read_project_tasks`, scoped to `<cwd>/tasks`), `GET /api/agents` -> all
  agents (filter by `project_id` client-side; there is no server-side per-project
  agents endpoint and none is needed).
- tatr feasibility: tasks are read per-project by filesystem cwd (not tagged by a
  project id anywhere). `/api/projects/{id}/tasks` is the right and only feed.
- The `/projects` page TODAY shows an INLINE detail panel on select (name, delete,
  cwd/language/description, tasks) - `projects-view.ts detailPanel` + the `select`
  action. The user wants a real PAGE, so this inline panel is SUPERSEDED and moves
  to `/projects/<id>` (same consolidation as agents: the list navigates, no inline
  detail).
- Shell pattern to mirror (per-agent): backend `_agent_detail_shell` + `GET
  /agents/{id}` + `GET /agents/{id}/{rest:path}` serving `agent-detail.html`
  (registered BEFORE the static mount; `/agents/` bare falls through to the static
  list). webpack: an entry + HtmlWebpackPlugin(filename `agent-detail.html`) + a
  dev rewrite `/^\/agents\/.+/`. Frontend: a shell html with `#header`/`#footer`
  partials, an entry that `initNav()`s + mounts, and an `idFromPath` helper.

## Steps (/plan)

- [ ] Backend (`app.py`): add `_project_detail_shell()` + `GET
      /projects/{project_id}` and `GET /projects/{project_id}/{rest:path}`
      (`include_in_schema=False`) serving `project-detail.html`, registered right
      after the agent-detail routes (before the static mount). 404 until built.
- [ ] webpack (`webpack.config.js`): add entry `"project-detail":
      "./src/project-detail.ts"`, an HtmlWebpackPlugin(template
      `./src/project-detail.html`, filename `project-detail.html`, chunk
      `project-detail`), and a dev rewrite `{ from: /^\/projects\/.+/, to:
      "/project-detail.html" }` placed BEFORE the bare `/^\/projects/` rewrite.
- [ ] `project-detail.html`: shell mirroring `agent-detail.html` - `#header`,
      a single `<main id="project-detail" class="settings">`, `#footer`.
- [ ] `project-detail.ts`: entry - `initNav()` + `startProjectDetail()`.
- [ ] `project-detail-view.ts`:
      - `projectIdFromPath(pathname)` -> id from `/projects/<id>`.
      - `renderProjectDetail(root, data, actions)` PURE: a back link to
        `/projects/`, the project name + a delete button, a metadata card
        (id/cwd/language/description), an Agents card (the project's agents, each
        a link to `/agents/<id>` with its state; empty state when none), and a
        Tasks card (the tatr tasks; loading/empty states) - reusing the existing
        `projtasks` markup.
      - `startProjectDetail()`: fetch the project (404 -> "no such project."),
        `/api/agents` filtered to `project_id === id`, and the tasks; wire delete
        to `DELETE /api/projects/<id>` then navigate to `/projects/`.
- [ ] `projects-view.ts`: make a registered project's name an `<a
      href="/projects/<id>">` (navigates to the page); REMOVE the inline
      `detailPanel`, the `select` action, and the `selectedId/selectedProject/
      tasks` state + their fetch orchestration. `renderProjects` simplifies to
      `(root, data, actions)`.
- [ ] Tests: backend - the two shell routes serve `project-detail.html`, a
      sub-path serves the same shell, `/projects/` (bare) still serves the static
      list, `/api/projects/{id}` is unaffected, and a missing frontend 404s
      (mirror `test_agent_detail_page_serves_shell` + `_404_without_frontend`).
      Frontend - `project-detail-view` renders metadata + the project's agents
      (filtered, linked) + tasks, shows the "no such project" fallback, escapes a
      hostile name, and delete navigates to `/projects/`; `projects-view` renders
      the project name as a link to `/projects/<id>` and no longer has an inline
      detail. Full suite (`ruff`/`mypy`/`pytest`, web `npm run ci`) green.

## Definition of Done

- Clicking a registered project on `/projects` navigates to `/projects/<id>`
  (test: the name is an anchor with that href; manual: the click opens the page).
- `/projects/<id>` shows the project's metadata, its registered agents (each
  linking to the agent's page), and its tatr tasks (test: the render + the shell
  route; manual: a real project renders all three).
- The bare `/projects` list still works and the inline detail panel is gone (test:
  `renderProjects` has no detail section; the static list route is not shadowed).
- Deleting from the detail page removes the project and returns to `/projects/`.
- Full backend + web suites green.
