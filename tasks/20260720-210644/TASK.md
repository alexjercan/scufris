# Projects backend: first-class Project store + CRUD API

- PRIORITY: 30
- TAGS: feature, projects, backend
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As the operator, I want projects to be first-class records - a workspace with a
name, language, description and cwd - so scufris can organize work by project
(the foundation for per-project agents/skills/tools in later phases). This task
delivers the store + CRUD API only; no agent wiring.

## Steps

- [x] Add `scufris/projects.py`: a `Project` pydantic model
      `{id, cwd, name, language, description}` and a `ProjectStore` that persists
      `projects.json` under `settings.state_dir` (atomic write like
      `settings_store._persist`; tolerate a missing/corrupt file on load).
      `id` is a URL-safe slug derived from the name (deduped with a numeric
      suffix on collision) - it is a path/URL segment, so validate it
      `^[A-Za-z0-9_-]+$` (fullmatch, not match+$).
- [x] Store methods: `list()`, `get(id)`, `create(name, cwd, language,
      description) -> Project`, `update(id, **fields) -> Project`, `delete(id)`.
      `create` validates: `name` non-empty, `cwd` is an existing directory
      (expand `~`); raise typed errors (`ProjectNotFound`, `InvalidProject`,
      `DuplicateProject`) the endpoint maps to status codes. Writes are gated by
      `settings.settings_writable` (raise the store's read-only error), matching
      the config store.
- [x] Add CRUD endpoints in `scufris/app.py`: `GET /api/projects` (list),
      `POST /api/projects` (create), `GET /api/projects/{id}` (one, 404),
      `PATCH /api/projects/{id}` (update, 404), `DELETE /api/projects/{id}`
      (404). Map errors: 403 read-only, 404 unknown, 409 duplicate, 422 invalid
      (bad name/cwd/slug). Inject the `ProjectStore` in `create_app` next to the
      `SettingsStore`.
- [x] Tests: store round-trip (create -> fresh store over same dir sees it;
      update/delete persist); create rejects a missing cwd (422) and empty name
      (422); duplicate name -> distinct deduped ids; endpoints return the right
      status codes incl. 403 when `settings_writable=false`.
- [x] Update `.env.example` only if a new knob is added (none expected -
      reuses `SCUFRIS_STATE_DIR`).

## Definition of Done

- A created project survives into a fresh `ProjectStore` over the same state dir;
  update and delete persist (test: `project_store_round_trip`).
- `create` with a non-existent cwd or empty name is rejected
  (test: `project_create_validates_cwd_and_name`).
- The CRUD endpoints return 200/404/409/422/403 correctly
  (test: `projects_crud_endpoints`, `projects_write_forbidden_when_readonly`).
- Full suite green (cmd: `nix develop --command bash -c "ruff check . && mypy . && python -m pytest -q"`).

## Notes

- Relevant files: `scufris/settings_store.py` (mirror its atomic-persist +
  read-only-gate + load-tolerance patterns), `scufris/config.py` (`state_dir`,
  `settings_writable`), `scufris/app.py` (`create_app`, model + endpoint style).
- Lessons: `fullmatch-not-match-dollar-for-id-validation` (the id is a URL
  segment); `in-place-mutation-beats-a-provider-rewire` is NOT needed here (the
  store is injected fresh, not captured in closures like settings).
- Run `python -m pytest` from the worktree (`nix-devshell-import-resolves-to-cwd-source`).
- Assumption: id is a name-derived slug (readable URLs); cwd is the natural
  identity but has slashes, so it is a field, not the key.

## Close-out

- Shipped `scufris/projects.py` (`Project` model + `ProjectStore` +
  ProjectNotFound/InvalidProject/DuplicateProject/ProjectsReadOnly) and the CRUD
  endpoints in `app.py`, injected next to the SettingsStore. Mirrored the
  settings store's atomic-persist + tolerant-load + read-only-gate patterns.
- id = a name-derived slug (`_slugify` -> `my-app`), deduped with a numeric
  suffix (`same`, `same-2`); validated fullmatch against PROJECT_ID_RE (it is a
  URL segment). cwd is a field (it has slashes), validated as an existing dir
  with `~` expansion.
- The ProjectStore is injected FRESH per create_app (not captured/mutated in
  closures like settings), so no in-place-mutation dance is needed - endpoints
  call the store directly.
- DuplicateProject is defined for completeness but the create path dedups ids
  rather than raising it (two projects can share a NAME, they get distinct ids);
  kept the 409 mapping for a future explicit-id create.
- 198 backend tests pass; ran `python -m pytest` from the worktree.
- Self-reflection: clean, pattern-following task - the settings store from the
  prior goal was a ready template, so this was mostly transcription with the
  slug/cwd-validation specifics.
