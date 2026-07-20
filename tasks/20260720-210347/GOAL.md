# Goal: projects orchestrator P0 - first-class Project entity (store + page + tatr view)

- DATE: 20260720
- UMBRELLA TASK: 20260720-210347
- LANDING SCOPE: squash-merge each task to `master` (local default), do NOT
  push (user's call). Standard flow landing.

## Goal

Phase P0 of the projects-orchestrator vision (tasks/20260720-184150/SPIKE.md
Revision 1): make a "project" a FIRST-CLASS entity instead of a bare cwd. A
project is a workspace record `{id, cwd, name, language, description}` persisted
in a scufris state file, with a CRUD API, a per-project tatr-tasks view (tatr is
directory-scoped), and a Projects PAGE in the web UI (list + create + a project
detail showing its metadata and its tatr tasks). This is the foundation the
later phases build on: per-project agent config (P1), multi-backend (P2),
per-project skills/tools (P3), the dispatch loop (P4).

This flips the earlier projects spike (tasks/20260720-182842) from its minimal
"cwd + {name, context_md}" (Option A) to a first-class object (Option B); its
seeded minimal tasks (182938/182953/182959) are re-cut into this P0 - the
session-scoping / per-turn `-C` parts of those defer to P1/P2 (they concern the
per-project AGENT, which P0 does not build).

Scope boundaries (P0 does NOT include): per-project agent config, driving an
agent scoped to a project, multi-backend, skills/tools, session<->project
grouping. P0 is the entity + its page + its tatr view only.

## Done means

1. A `ProjectStore` persists projects to a state file and round-trips
   (create/list/get/update/delete survive a fresh store over the same dir).
   (test: `project_store_round_trip`)
2. CRUD API: `GET /api/projects`, `POST /api/projects`, `GET /api/projects/{id}`,
   `PATCH /api/projects/{id}`, `DELETE /api/projects/{id}` - create validates
   (existing dir, non-empty name), 404 on unknown id, gated by
   settings_writable for mutations. (test: endpoint CRUD + validation tests)
3. A per-project tatr view: `GET /api/projects/{id}/tasks` returns the project's
   tatr tasks by running tatr scoped to the project's cwd; empty (not an error)
   when the dir has no tasks. (test: `project_tasks_endpoint` against a temp
   project dir with tatr tasks)
4. A Projects PAGE: lists projects, a create form, and a project detail view
   showing name/language/description/cwd and the project's tatr tasks. Serves at
   `/projects/`. (cmd: `cd web && npm run ci`; manual: the page lists/creates a
   project and shows its tatr tasks on the running app)

Overall: the full check suite passes on master (cmd: `nix develop --command
bash -c "ruff check . && mypy . && pytest -q"` plus `npm run ci` in web/), and
`tatr check --ledger LESSONS.md` is clean for this goal's tasks.

## Tasks

Updated as tasks land (one line per land). Order = priority; dependencies noted.

- [x] 20260720-210644 (p30) Projects backend: first-class Project store + CRUD API
      landed 35d6cc4; 1 review round (APPROVE, 2 NITs fixed); ProjectStore + CRUD, slug ids, cwd validation
- [ ] 20260720-210645 (p28) Projects backend: per-project tatr-tasks endpoint [dep: 210644]
- [ ] 20260720-210647 (p25) Projects UI: Projects page (list + create + detail with tatr tasks) [dep: 210644, 210645]

## Manual acceptance (batched for the user at Finish)

Accumulates `manual:` DoD items as tasks land; presented at Finish.

(none yet)
