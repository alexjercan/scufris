# Review: Refresh the scufris-web npmDepsHash

- TASK: 20260803-022030
- BRANCH: master

## Round 1

- REVIEWER: maintainer
- VERDICT: APPROVE

One line. `flake.nix:132` takes the hash Nix itself printed, and
`nix build .#scufris-web` goes from a fixed-output mismatch to exit 0.

The cause was the first hypothesis in the record: `web/package-lock.json` moved
in `e816f46` (2026-08-02) and the pin, last set in `f7e44c7` (2026-07-21), did
not follow. Verified by the two commit dates rather than assumed.

No findings. One thing noted and deliberately left: nothing ties the lockfile to
the pin, so the next `package-lock.json` change reddens the build the same way.
A guard belongs with the next frontend change, not inside a one-line fix.
