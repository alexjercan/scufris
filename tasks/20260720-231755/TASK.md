# fix nix flake check: pytest derivation cannot import scufris

- STATUS: OPEN
- PRIORITY: 0
- TAGS: backlog,bug

## Story

As a scufris developer, I want `nix flake check`'s pytest derivation to import
scufris, so that the QA gate is green and can be used as the goal green bar.
Currently it fails with `ModuleNotFoundError: No module named 'scufris'` (the
mkCheck pytest sandbox does not install/expose the package), so `nix flake check`
is red on master while the devShell `python -m pytest` passes (203).

## Steps

- [ ] Reproduce: `nix flake check` and confirm the scufris-pytest derivation fails with ModuleNotFoundError.
- [ ] Diagnose mkCheck in flake.nix: the pytest check runs in a sandbox that lacks the editable/installed scufris on sys.path (unlike the devShell venv).
- [ ] Fix so the pytest check imports scufris (install the package into the check env, or set PYTHONPATH/REPO_ROOT equivalently in mkCheck), mirroring how ruff/mypy see the tree.
- [ ] Confirm `nix flake check` is fully green.

## Definition of Done

- `nix flake check` passes including the pytest derivation (cmd: `nix flake check`).

## Notes

- Discovered during the flow-guards goal (20260720-225502); confirmed pre-existing on master (baseline-dod-proofs).
- The devShell path already works (`nix develop -c python -m pytest` = 203 passed); this is only the sandbox check.
