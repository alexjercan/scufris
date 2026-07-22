# project detail page /projects/<id> - tasks, registered agents, metadata

- STATUS: OPEN
- PRIORITY: 40
- TAGS: projects,frontend,backend

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
