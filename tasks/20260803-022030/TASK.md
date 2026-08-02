# Refresh the scufris-web npmDepsHash

- PRIORITY: 45
- TAGS: bug,nix,web
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -

## Story

As a maintainer, I want `nix build .#scufris-web` to succeed on master, so that
the frontend package - one of the two builds CI calls green - is buildable.

## Steps

- [ ] Reproduce: `nix build .#scufris-web` fails with a fixed-output hash
      mismatch on `scufris-web-0.1.0-npm-deps`.
- [ ] Work out whether `web/package-lock.json` moved without `npmDepsHash`
      following it, or whether the pin needs refreshing for another reason.
- [ ] Update the hash in `nix/` and confirm the build.

## Definition of Done

- The frontend package builds (cmd: `nix build .#scufris-web`).

## Notes

- Found while working 20260729-102148; reproduces on a clean master checkout, so
  it is not that branch's doing.
- specified: `sha256-KncgMKbpFwCIEYeSIcqddfXutzFnY0EMcnaT+bK0WZU=`
- got: `sha256-ZbmYSEmFsJdaSMEItWwdJE5yl1Lf7paBvtSaxak6eRI=`
