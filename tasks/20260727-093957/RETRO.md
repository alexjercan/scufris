# Retro: nvidia-smi on the deployed scufris service PATH

- TASK: 20260727-093957
- BRANCH: (scufris) bug/gpu-service-path; (nix.dotfiles) feature/scufris-gpu-service-path
- REVIEW ROUNDS: 1 (out-of-context; APPROVE with one MEDIUM addressed)

## What went well

- Diagnosis was fast and evidence-first: read the collector (`shutil.which`),
  the HM service module (PATH override), and the live rendered unit PATH, then
  confirmed against the running service's actual `Environment=PATH=`. No
  guessing about why local worked and deployed did not.
- Nailed the crux (NVML version match) BEFORE building by `nix eval`-ing
  `pkgs.linuxPackages.nvidia_x11.bin` and comparing its store path to
  `readlink -f /run/current-system/sw/bin/nvidia-smi` - identical. That single
  check turned "probably matches" into proof and de-risked the whole fix.
- Reused the today/macros deploy playbook exactly: add the bin package to
  `programs.scufris.path`, prove via `activationPackage` build + rendered-unit
  grep (lesson `render-hm-unit-file-not-eval`).
- Added an end-to-end proof beyond PATH rendering: ran the app's exact
  `_GPU_QUERY` through that store path's nvidia-smi and got valid 9-field CSV.

## What went wrong

- First rendered-unit grep used a wrong path (`find ./result -name scufris.service`
  returned nothing because the unit lives under
  `result/home-files/.config/systemd/user/`). Cost one round-trip. The
  `render-hm-unit-file-not-eval` lesson says BUILD the activationPackage but does
  not record WHERE the user unit renders; noted below.
- The first comment framed the version-match guarantee as "same nixpkgs input,"
  which the reviewer correctly flagged (MEDIUM) as understating the real
  dependency: it also needs the host on the DEFAULT kernel with
  `nvidiaPackages.stable`. Fixed the comment to state that assumption and the
  escape hatch. Lesson: when a nix equivalence is load-bearing, name the exact
  conditions it depends on, not just the shared input.

## What to improve next time

- Cross-repo nix.dotfiles change again touched the user's actively-used repo;
  followed `recheck-head-before-committing-in-a-user-touched-repo` (re-checked
  HEAD = master 11ad4a6, clean) and worked on a feature branch. Keep doing this.
- Record the HM user-unit render path in the ledger so the next unit-grep does
  not fumble the find.

## Action items

- [x] scufris: task record only, landed via sprout on bug/gpu-service-path.
- [ ] nix.dotfiles: `feature/scufris-gpu-service-path` holds the one-line path
      fix + comment; merge to master + `home-manager switch` is the operator's
      call (flow does not push/deploy). GPU appears after the switch.
- [ ] Ledger: add `hm-user-unit-renders-under-home-files-systemd-user` and
      `name-the-conditions-a-nix-equivalence-depends-on`.
