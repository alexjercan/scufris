# Review: Projects UI - Projects page (list + create + detail with tatr tasks)

- TASK: 20260720-210647
- BRANCH: feature/projects-page

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context (fresh subagent; recreated the node_modules symlink,
  ran full `npm run ci`, removed the symlink after)

No BLOCKER/MAJOR. `npm run ci` green (lint + tsc + 125 vitest across 6 files +
webpack build; all four pages emit incl. `projects/index.html`). Reviewer
verified XSS escaping on every host/user string (name/desc/cwd/language, task
title/tags; list name via textContent), `renderProjects` purity + lazy task
load with a loading state degrading to empty (not blank), the create form
requiring name+cwd and clearing on success, single-authoritative reload after
mutations, stale-selection drop after delete, and consistent multipage wiring
(entry + plugin + rewrite + nav link + initNav).

- [x] R1.1 (MINOR) projects-view.ts `startProjects.select` - the tasks fetch
  wrote the shared `tasks` var without checking the selection was still current,
  so rapidly selecting A then B could render A's tasks under B.
  - Response: fixed - guarded all three writes/render (`then`/`catch`/`finally`)
    with `if (selectedId === id)`, so a stale response for a previous selection
    is dropped. CI re-run green.

Open `manual:` DoD item (batched to the user at Finish): load `/projects/`,
create a project pointing at a dir with tatr tasks, and see them - confirmed via
the /work e2e serve (page 200, create -> id=demo, the demo project's real p30
tatr task shown, nav link present).
