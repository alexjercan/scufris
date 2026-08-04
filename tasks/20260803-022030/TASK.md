# Refresh the scufris-web npmDepsHash

- PRIORITY: 45
- TAGS: bug, v0.2.0, nix, web
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Story

As a maintainer, I want `nix build .#scufris-web` to succeed on master, so that
the frontend package - one of the two builds CI calls green - is buildable.

## Steps

- [x] Reproduce: `nix build .#scufris-web` fails with a fixed-output hash
      mismatch on `scufris-web-0.1.0-npm-deps`.
- [x] Work out whether `web/package-lock.json` moved without `npmDepsHash`
      following it, or whether the pin needs refreshing for another reason.
- [x] Update the hash in `nix/` and confirm the build.

## Definition of Done

- The frontend package builds (cmd: `nix build .#scufris-web`).

## Notes

- Found while working 20260729-102148; reproduces on a clean master checkout, so
  it is not that branch's doing.
- specified: `sha256-KncgMKbpFwCIEYeSIcqddfXutzFnY0EMcnaT+bK0WZU=`
- got: `sha256-ZbmYSEmFsJdaSMEItWwdJE5yl1Lf7paBvtSaxak6eRI=`
- Cause (2026-08-03): the first hypothesis. `web/package-lock.json` last moved
  in `e816f46` (2026-08-02, "make the ports on DEV not hardcoded"); the hash was
  last set in `f7e44c7` (2026-07-21). The lockfile moved and the pin did not
  follow it.
- The hash lives at `flake.nix:132`, not under `nix/` as the third step assumed.
- This is a standing trap, not a one-off: nothing ties the two together, so the
  next `package-lock.json` change reddens the build again the same way. Worth a
  guard when the frontend is next touched - out of scope for a one-line fix.
