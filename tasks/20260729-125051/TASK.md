# Add continuous integration for every push and pull request

- STATUS: OPEN
- PRIORITY: 100
- TAGS: infra,v0.1.0,ci,nix,frontend

## Story

As the maintainer, I want every push and pull request checked automatically, so
that the QA gate stops depending on remembering to run it and a broken master is
noticed by the repository rather than by the next session.

The gate already exists and is well defined: `nix flake check` runs ruff, mypy,
and pytest; `cd web && npm run ci` runs prettier, eslint, vitest, and the
webpack build. Neither has ever run outside a developer's shell.

## Steps

- [ ] Add `.github/workflows/ci.yaml` triggered on push to master and on pull
      requests, with a concurrency group that cancels superseded runs on the
      same ref.
- [ ] Install Nix on the runner and make the build affordable: pick the
      installer action and the binary cache strategy, and record what a cold
      uv2nix build costs before caching. If a full `nix flake check` is not
      viable on a hosted runner, say so in the task record and run the venv
      path directly instead of quietly skipping checks.
- [ ] Run the Python gate: ruff, mypy, and pytest, through the same entry point
      the developer shell uses so the two cannot drift.
- [ ] Run the frontend gate: `npm ci` then `npm run ci` (format check, lint,
      vitest, production build).
- [ ] Run repository conformance: `tatr check --ledger LESSONS.md`, so task
      records and the lessons ledger are enforced mechanically instead of by
      session discipline.
- [ ] Set a job timeout, and make failures readable: name the steps after what
      they check, never pipe a gate command into something that eats its exit
      code.
- [ ] Add a status badge and document in AGENTS.md that CI is the source of
      truth for the gate.

## Definition of Done

- A push to master and a pull request both run the full gate on a clean checkout
  (cmd: `gh run list --workflow ci --limit 5`).
- A deliberately broken lint, type, test, frontend, or task record fails CI
  (manual: verified once on a scratch branch, recorded in the task).
- The workflow's steps match the gate AGENTS.md documents, with no silently
  skipped check (cmd: `rg -n "ruff|mypy|pytest|npm run ci|tatr check" .github/workflows/ci.yaml`).
- Total runtime on a warm cache is low enough that it does not discourage small
  pushes (manual: recorded timing in the task).

## Notes

- Epic: 20260729-124706.
- Reference: `~/personal/nova-protocol/.github/workflows/ci.yaml` - concurrency
  cancellation, explicit toolchain install, cache-on-failure, and comments that
  explain WHY each step exists.
- Repository: `git@github.com:alexjercan/scufris.git`.
- `nix build .#vm-test` (the NixOS VM test) is deliberately outside `checks`
  because it needs KVM. Decide here whether it belongs in CI at all, or only in
  the release gate, and write the reason down.

## Flow State

- FLOW STEP: PLANNING
