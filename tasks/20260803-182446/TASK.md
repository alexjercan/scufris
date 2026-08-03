# `nix flake check` does not check Python formatting

- PRIORITY: 30
- TAGS: chore, tooling
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

As a contributor, I want the repository's own gate to fail on a formatter
violation, so that a badly formatted line cannot reach review the way
`scufris/telegram/render.py`'s 89-char line did on
`chore/diagnostics-minors`.

## Notes

`flake.nix:233` runs `ruff check .` only. `AGENTS.md:53` names `ruff format .`
as a required command, but nothing enforces it: `ruff format --check .` is
never run by the gate. A branch can therefore be green under `nix flake check`
while `ruff format --check .` reports a file would be reformatted - which is
exactly what happened, and it took a human reviewer to catch.

Found while addressing round-1 review feedback on 20260803-042958. Pre-existing
and separable: it is a change to the flake, not to any surface that task
touches.

Likely shape: add a `ruff format --check .` step to the existing ruff check
derivation, or a sibling check next to it. Confirm the web side too -
`npm run ci` does include `format:check`, so only Python is unguarded.
