# Add continuous integration for every push and pull request

- STATUS: CLOSED
- PRIORITY: 100
- TAGS: infra,v0.1.0,ci,nix,frontend
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

As the maintainer, I want every push and pull request checked automatically, so
that the QA gate stops depending on remembering to run it and a broken master is
noticed by the repository rather than by the next session.

The gate already exists and is well defined: `nix flake check` runs ruff, mypy,
and pytest; `cd web && npm run ci` runs prettier, eslint, vitest, and the
webpack build. Neither has ever run outside a developer's shell.

## Steps

- [x] Add `.github/workflows/ci.yaml` triggered on push to master and on pull
      requests, with a concurrency group that cancels superseded runs on the
      same ref.
- [x] Install Nix with `DeterminateSystems/nix-installer-action` and run the
      real `nix flake check`, relying on `cache.nixos.org` with no third-party
      cache (DECISION.md). Record the measured cold and warm wall-clock cost in
      this task. If a full `nix flake check` turns out not to be viable on a
      hosted runner, say so in the task record and run the venv path directly
      instead of quietly skipping checks.
- [x] Run the Python gate: ruff, mypy, and pytest, through the same entry point
      the developer shell uses so the two cannot drift.
- [x] Run the frontend gate: `npm ci` then `npm run ci` (format check, lint,
      vitest, production build).
- [x] Run repository conformance: `tatr check --ledger LESSONS.md`, so task
      records and the lessons ledger are enforced mechanically instead of by
      session discipline.
- [x] Set a job timeout, and make failures readable: name the steps after what
      they check, never pipe a gate command into something that eats its exit
      code.
- [x] Add a status badge and document in AGENTS.md that CI is the source of
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
  because it needs KVM. It stays out of THIS workflow: it guards the release
  only (see `tasks/20260729-125101/DECISION.md`). Say so in the workflow's
  comments so its absence reads as a decision, not an oversight.
