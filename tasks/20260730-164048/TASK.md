# Namespace every flake output with a scufris prefix

- STATUS: CLOSED
- PRIORITY: 30
- TAGS: refactor,nix,packaging
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

As the operator pinning this flake from a NixOS/home-manager config alongside
other flakes, I want every Scufris output named with a `scufris` prefix, so that
`packages.web` or `nixosModules.hostd` cannot be confused with (or shadowed when
re-exported next to) another project's outputs. Today the app package and the
option paths are namespaced (`packages.scufris`, `services.scufris`,
`services.scufris-hostd`) but the other outputs are not.

This is a pure rename of the flake's public attribute surface plus every
reference to it. No derivation, module, option path, or behavior changes.

## Rename map

| old | new |
|-----|-----|
| `packages.web` | `packages.scufris-web` |
| `packages.vm-test` | `packages.scufris-vm-test` |
| `packages.hostd-vm-test` | `packages.scufris-hostd-vm-test` |
| `nixosModules.hostd` | `nixosModules.scufris-hostd` |
| (new) | `nixosModules.scufris` (same module as `nixosModules.default`) |
| (new) | `homeManagerModules.scufris` (same module as `homeManagerModules.default`) |

Unchanged, decided at the gate:

- `packages.default`, `apps.default`, `nixosModules.default`,
  `homeManagerModules.default` STAY, so `nix build .` / `nix run .` and any
  existing `nixosModules.default` import keep working.
- `packages.scufris` and `apps.scufris` are already namespaced.
- `checks.{ruff,mypy,pytest,records}` keep their short names: they are labels
  for `nix flake check`, and their derivations are already `scufris-*` via
  `mkCheckWith`.
- Old names are REMOVED, not aliased (clean break).

## Steps

- [x] Rename the outputs in `flake.nix`: the three `packages` attrs, the
      `nixosModules.hostd` attr, and add the `scufris` aliases for the nixos and
      home-manager modules. Keep the `default` attrs. Update the internal
      references too (`self.nixosModules.hostd` feeding the hostd VM test, and
      the `nix build .#vm-test` comment on the `optionalAttrs` block).
- [x] Update the two CI workflows: `.github/workflows/ci.yaml` and
      `release.yaml` build `.#web`, `.#vm-test` and `.#hostd-vm-test`.
- [x] Update the nix sources' own comments that name outputs:
      `nix/scufris-service.nix` (`packages.web`), `nix/scufris-hostd.nix`
      (`nixosModules.hostd`), `nix/tests/scufris-vm.nix`
      (`nixosModules.default`, `packages.web`). This step was written as
      comments-only and was WRONG: `nix/scufris-service.nix` also RESOLVES the
      package (`defaults.web`, line 63, plus its `defaultText`), so the rename
      was load-bearing code there, not prose. `nix flake check` caught it.
- [x] Sweep the live doc surfaces named by AGENTS.md: root `README.md` (the
      outputs table and every `nix build` line), `AGENTS.md`, `scufris/README.md`
      (the commands table), `scufris/hostd/README.md` (the `imports = [...]`
      snippet and the VM-test command), `web/README.md` (`packages.web`), and
      `examples/nixos_change.py` (`.#hostd-vm-test` in its docstring). The
      command blocks in `README.md`, `AGENTS.md` and `scufris/hostd/README.md`
      were re-aligned: the longer names pushed their `#` comment columns out.
- [x] Add a CHANGELOG `### Changed` entry under Unreleased that gives the full
      old -> new mapping, since this breaks any consumer pinning the old names.

## Definition of Done

- Every new name evaluates to a module or a derivation:
  cmd: `nix eval --raw .#packages.x86_64-linux.scufris-web.outPath`
  cmd: `nix eval .#nixosModules.scufris-hostd --apply 'm: builtins.isFunction m || builtins.isAttrs m'`
  cmd: `nix eval .#nixosModules.scufris --apply 'm: builtins.isFunction m || builtins.isAttrs m'`
  cmd: `nix eval .#homeManagerModules.scufris --apply 'm: builtins.isFunction m || builtins.isAttrs m'`
  (the plan said `builtins.isAttrs`; that only holds for the NixOS attrs, which
  flake-parts coerces through its `deferredModule` option type. The
  home-manager attr is the raw module FUNCTION, so `isAttrs` is false there and
  is the wrong probe.)
- The old names are gone: `nix eval .#packages.x86_64-linux.web` and
  `nix eval .#nixosModules.hostd` both FAIL.
- The `default` attrs still resolve:
  cmd: `nix eval --raw .#packages.x86_64-linux.default.outPath`
  cmd: `nix eval .#nixosModules.default --apply 'm: builtins.isFunction m || builtins.isAttrs m'`
  cmd: `nix eval .#homeManagerModules.default --apply 'm: builtins.isFunction m || builtins.isAttrs m'`
- No live surface still names an old output: that grep is run over EVERY tracked
  file, not a hand-listed set of docs -
  `git grep -nE '\.#web|\.#vm-test|\.#hostd-vm-test|packages\.web|nixosModules\.hostd'`
  returns nothing outside `tasks/`, `LESSONS.md` and the CHANGELOG entries for
  already-released versions (plus the new mapping table that translates them).
  Review round 1 found `.gitignore` had a `nix build .#scufris .#web` comment,
  which the doc-surface list did not cover: a tracked dotfile carrying a command
  IS a surface.
- The shipped packages build under their new names:
  cmd: `nix build --no-link .#scufris .#scufris-web`
- cmd: `nix flake check`
- cmd: `nix build --no-link .#scufris-vm-test` (needs KVM; this is what proves
  `nixosModules.scufris` still builds a booting unit whose `/` is served from
  the renamed `packages.scufris-web`)
- cmd: `nix build --no-link .#scufris-hostd-vm-test` (needs KVM; proves
  `nixosModules.scufris-hostd` still activates on a real root socket)

## Notes

- `tasks/` records and `LESSONS.md` entries are append-only history and are NOT
  swept: they keep the names that were true when written (AGENTS.md, Docs sync).
  The CHANGELOG mapping is what lets a reader translate an old entry.
- `nix build .#vm-test` was never in CI (needs KVM); it runs in the release
  workflow, so the rename must land in `release.yaml` or the release pipeline
  breaks on the next tag.
- `nixosModules.scufris` and `nixosModules.default` are ONE module value (a
  single `let` binding), so importing both cannot declare `services.scufris`
  twice. This is not provable with `==`: Nix compares functions (and attrsets
  containing them) as unequal, so both `m.scufris == m.default` probes return
  false regardless. Construction is the proof; the flake comment states it.
- `cd web && npm run ci` was NOT run: no file under `web/src` changed, and the
  frontend gate's prettier scope is `src/**` plus the configs, so `web/README.md`
  is outside it.
