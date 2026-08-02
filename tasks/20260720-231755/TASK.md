# fix nix flake check: pytest derivation cannot import scufris

- PRIORITY: 0
- TAGS: backlog, bug
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As a scufris developer, I want `nix flake check`'s pytest derivation to import
scufris, so that the QA gate is green and can be used as the goal green bar.
Currently it fails with `ModuleNotFoundError: No module named 'scufris'` (the
mkCheck pytest sandbox does not install/expose the package), so `nix flake check`
is red on master while the devShell `python -m pytest` passes (203).

## Steps

- [x] Reproduced: `nix flake check` and confirm the scufris-pytest derivation fails with ModuleNotFoundError.
- [x] Diagnosed mkCheck in flake.nix: the pytest check runs in a sandbox that lacks the editable/installed scufris on sys.path (unlike the devShell venv).
- [x] Fixed: mkCheck runs `python -m pytest` (prepends cwd) so scufris imports from the copied tree. Surfaced+fixed 2 more sandbox issues: fake codex/appserver scripts now resolve their interpreter (not `/usr/bin/env`, absent in the sandbox); the 7 real-tatr integration tests are gated `@pytest.mark.needs_tatr` and skipped when tatr is off PATH (user chose skip over bundling).
- [x] Confirmed `nix flake check` -> all checks passed (232 pass + 7 skip); devShell runs all 239.

## Definition of Done

- `nix flake check` passes including the pytest derivation (cmd: `nix flake check`).

## Notes

- Discovered during the flow-guards goal (20260720-225502); confirmed pre-existing on master (baseline-dod-proofs).
- The devShell path already works (`nix develop -c python -m pytest` = 203 passed); this is only the sandbox check.
