# Refresh project documentation and baseline browser polish

- STATUS: OPEN
- PRIORITY: 0
- TAGS: docs,backlog,ui,frontend

## Story

As a new user or developer, I want the repository documentation and browser
shell to match the application that exists today, so that setup instructions,
project orientation, and first impressions are trustworthy.

## Steps

- [ ] Update README and AGENTS orientation from the original spike-era state to
      the implemented monitoring, agent, project, and Telegram surfaces.
- [ ] Correct test commands, project layout, lessons location, frontend QA
      commands, mock-demo behavior, and localhost-only security assumptions.
- [ ] Update stale package descriptions and examples against actual routes and
      configuration.
- [ ] Add an application favicon and verify every page references an available
      asset.
- [ ] Resolve the current Starlette/httpx and asyncio deprecation warnings or
      pin/document an upstream blocker with a follow-up task.
- [ ] Add a small documentation consistency test for commands and referenced
      repository paths.

## Definition of Done

- The documented quickstart starts the current application and uses
  `python -m pytest` (test: `test_readme_commands_and_paths_are_current`).
- README contains no stale statement that the lessons ledger lives under
  `docs/` (test: `test_readme_commands_and_paths_are_current`).
- Browser route smoke produces no favicon `404`
  (test: `route-smoke.spec.ts`).
- The default Python suite emits no known deprecation warning
  (cmd: `python -m pytest -W error`).

## Notes

- Epic: 20260729-102149.
- Depends on: 20260729-102152 for the browser asset assertion.
- Keep documentation changes synchronized with any QA commands finalized by
  20260729-102154.

## Flow State

- FLOW STEP: PLANNING
