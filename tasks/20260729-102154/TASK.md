# Add frontend and browser suites to canonical QA gates

- STATUS: OPEN
- PRIORITY: 65
- TAGS: infra, v0.2.0, testing, nix, frontend

## Story

As a maintainer, I want the documented quality gate to cover the complete
shipped application, so that frontend or browser regressions cannot land behind
a Python-only green `nix flake check`.

## Steps

- [ ] Add named package scripts for formatting checks, lint, unit tests,
      production build, browser smoke, and the complete frontend suite.
- [ ] Package the deterministic frontend checks as Nix derivations using locked
      dependencies and no mutable `node_modules` leakage.
- [ ] Decide and record which browser checks are light enough for every flake
      check and which belong in an explicit full/release gate.
- [ ] Wire the selected checks into `flake.nix`, preserving useful separate
      derivation names and failure output.
- [ ] Add one documented command that runs all release proofs, including the
      NixOS VM deployment test.
- [ ] Verify the gate fails when a controlled frontend test or build fixture is
      broken, then remove the fault.

## Definition of Done

- `nix flake check` covers Ruff, mypy, pytest, frontend format/lint/unit tests,
  and the production web build (cmd: `nix flake check`).
- The named full gate runs browser journeys and the NixOS VM deployment test
  (cmd: `./scripts/qa-full`).
- The README and AGENTS commands agree with the implemented gates
  (test: `test_documented_qa_commands_exist`).
- No pipeline masks an earlier command's failure status
  (cmd: `! rg -n '\\| *(grep|tail|head)|; *echo' flake.nix scripts/ web/package.json`).

## Notes

- Epic: 20260729-102149.
- Depends on: 20260729-102152 and 20260729-102153.
- V0.2.0 readiness role: make the browser harness an enforced landing gate
  before the cross-page orchestrator feature starts.
- Record the fast-gate versus full-gate decision in `DECISION.md`.

## Flow State

- FLOW STEP: PLANNING
