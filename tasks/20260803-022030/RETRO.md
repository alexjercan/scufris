# Retro: Refresh the scufris-web npmDepsHash

- TASK: 20260803-022030
- BRANCH: master
- REVIEW ROUNDS: 1

## What went well

- The tool handed over the fix. Nix prints both hashes on a fixed-output
  mismatch, so reproducing the failure and learning the answer were the same
  step.
- The record's own first hypothesis was right and was checked rather than
  trusted: two `git log` calls on `web/package-lock.json` and on the hash line
  settled it in seconds.

## What went wrong

- This sat red on master long enough to be found while working an unrelated
  task, and then scheduled as its own record. A build that CI calls green was
  broken by a commit that had nothing to do with packaging, and nothing noticed.
- The record's third step said the hash lives under `nix/`. It is at
  `flake.nix:132`. Harmless here, but the step was written from memory of the
  tree rather than from the tree.

## What to improve next time

- Check the canonical gates before planning a sprint on top of them. This was
  found during pre-flight for the v0.2.0 carve, which is late but not too late;
  finding it after the carve started would have made every failure ambiguous.
- When a record names a file path, cite it with a line number so the step is
  falsifiable at read time.

## Action items

- [ ] Tie `npmDepsHash` to `web/package-lock.json` so the two cannot drift
      silently - a check that fails loudly beats a build that fails cryptically.
      Belongs with the next frontend change, not here.
