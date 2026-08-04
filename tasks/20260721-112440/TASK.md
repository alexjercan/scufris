# B6: sesh.py directory discovery + Projects discovery/create (no tmux)

- PRIORITY: 25
- TAGS: agents, backend, projects
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

`scufris/sesh.py`: `discover()` scans configurable base dirs one level deep ->
candidate {path, name, language?} (language inferred from marker files:
pyproject.toml->python, package.json->node, Cargo.toml->rust, ...); `create(name,
base)` -> mkdir (NO tmux) and returns the path. Projects page surfaces DISCOVERED
dirs UNION registered projects (marking which are registered); create registers +
mkdirs. Base dirs default to the sesh set (~/personal, ~/personal/_tests, ~/work,
~/third-party), configurable in settings.

## Steps (/plan)

- [x] `scufris/sesh.py`: `Candidate {path, name, language}`; `MARKERS` map
      (pyproject.toml->python, package.json->node, Cargo.toml->rust, go.mod->go,
      ...); `infer_language(dir)`; `discover(base_dirs)` scans each base ONE level
      deep (dirs only, skip hidden), dedups by path, sorted by name;
      `create(name, base)` mkdirs `base/<safe-name>` (NO tmux, rejects traversal),
      returns the path. Pure/deterministic - unit-tested with tmp dirs.
- [x] `config.py`: `project_base_dirs: list[Path]` default the sesh set
      (~/personal, ~/personal/_tests, ~/work, ~/third-party), env
      `SCUFRIS_PROJECT_BASE_DIRS` (colon-separated or JSON) via a before-validator.
- [x] `app.py`: `GET /api/projects/discovered` -> discovered UNION registered,
      each `{path, name, language, registered, project_id}` (registered dirs
      marked, discovered-only ones not); `POST /api/projects/new {name, base}` ->
      mkdir under an ALLOWED base (422 otherwise) + register + return the Project.
      Registering an already-existing discovered dir uses the existing
      `POST /api/projects`.
- [x] Frontend `projects-view.ts` + `common.ts`: list discovered UNION registered,
      badge which are registered; a "register" action for a discovered dir and a
      "create" form (name + base picker from the discovered bases). Port tests.
- [x] Full check suite green (backend pytest/ruff/mypy + web `npm run ci`).

## Definition of Done

- `discover()` finds one-level-deep dirs under the base dirs and infers language
  from marker files; files and hidden dirs are skipped
  (test: `tests/test_sesh.py`).
- `create(name, base)` makes the directory and NO tmux/subprocess is spawned
  (test: asserts the dir exists; a traversal name is rejected).
- `GET /api/projects/discovered` returns discovered UNION registered with a
  `registered` flag; `POST /api/projects/new` mkdirs under an allowed base and
  registers, 422 for a base outside `project_base_dirs`
  (test: `tests/test_app.py`).
- Projects page shows discovered + registered dirs and can create/register one
  (test: `web/src/projects-view.test.ts`).
- Full check suite green.
- manual: the Projects page lists my real dirs and creating one works end to end.

## Notes
- Spike: tasks/20260721-112212/SPIKE.md (EPIC 20260721-112212) (decision 4; recommendation B6/F5). NO tmux - directory only.
- Independent; can slot anywhere.
