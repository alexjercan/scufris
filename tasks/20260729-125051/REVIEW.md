# Review: Add continuous integration for every push and pull request

- DATE: 20260729-134229
- ROUND: 1
- REVIEWER: out-of-context agent
- VERDICT: REQUEST_CHANGES

## Findings

### MAJOR Definition of Done not met: no measured timings, no red-CI evidence, no NOTES.md in the task record

`tasks/20260729-125051/TASK.md:19-27` - step 2 is still `[ ]` and says in its own
words "Record the measured cold and warm wall-clock cost in this task". The task
record contains no timing at all. The fourth DoD item ("Total runtime on a warm
cache is low enough ... manual: recorded timing in the task") and the second DoD
item ("A deliberately broken lint ... fails CI (manual: verified once on a
scratch branch, recorded in the task)") both require the evidence to live IN the
record, and it does not. The evidence exists (run 30443720539 succeeded at
nix 1m51s / web 32s; run 30443929343 failed BOTH jobs on a ruff + prettier
break; run 30444113311 succeeded after the revert - I verified all three with
`gh run view`), but a cold session reading `tasks/20260729-125051/` learns none
of it. The whole point of the DECISION.md is that the timing becomes the
argument for or against a follow-up binary-cache task, and that argument is
currently unwritten.

Also missing: `tasks/20260729-125051/NOTES.md`. Repo AGENTS.md ("Where records
go") lists NOTES.md as the design/fix record for a shipped change, and the
global `~/AGENTS.md` requires a written record of what changed and why,
difficulties hit, and self-reflection. Nothing was written.

Suggested change: tick step 2, and add a "Verification" section to TASK.md (or a
NOTES.md) with the three run IDs, the per-job wall-clock numbers, what the
deliberate break touched (`scufris/health.py` ruff + `web/src/common.ts`
prettier), the fact that the `records` check was proven red only locally by
corrupting a FLOW STEP value (never red in CI), and the observation that on a
hosted runner cold and warm are effectively the same (~2 min) because nothing
persists the /nix/store between runs - there is no nix cache action, so
`cache.nixos.org` is the only warmth there is. State plainly whether that
closes or re-opens the binary-cache question.

### MAJOR Workflow grants the default GITHUB_TOKEN permissions

`.github/workflows/ci.yaml:1-19` - there is no `permissions:` block at workflow
or job level, so both jobs get whatever the repository/organization default is,
which on older repos is read/write on `contents`, `issues`, `pull-requests`,
`packages`, etc. Neither job needs any token scope beyond reading the checkout.
This matters more than usual here because the `nix` job runs arbitrary build
code from the PR head (`nix flake check` evaluates and executes the PR's own
`flake.nix`), so a malicious or compromised PR would execute with whatever token
the runner was handed. Fork PRs get a read-only token regardless, but a
same-repo branch does not.

Suggested change: add at the top of the workflow, above `jobs:`:

```yaml
permissions:
  contents: read
```

### MAJOR Third-party action pinned to a mutable branch

`.github/workflows/ci.yaml:34` - `uses: DeterminateSystems/nix-installer-action@main`.
This is a floating branch ref of a third-party action that installs a daemon as
root on the runner. Any push to that repository's `main` changes what runs in
this repository's CI, with no review and no lockfile entry. That is exactly the
supply-chain hole the DECISION.md's "no secrets, no operator setup" reasoning
does not cover. It is also inconsistent with the flake change in the same diff,
whose comment argues that `tatr` is an input precisely so it "cannot start
failing here because someone pushed to tatr's master" - the same argument
applies verbatim to the installer action.

Suggested change: pin to a release tag (e.g. `@v20`) or, better, a commit SHA
with a trailing `# v20.x` comment. Same treatment is worth considering for
`actions/checkout@v4` and `actions/setup-node@v4`, though major-version tags
from the `actions` org are a defensible middle ground; the `@main` on a
third-party root-installing action is not.

### MINOR `nix flake check` never builds the package outputs, so `.#scufris` and `.#web` can be broken while CI is green

`flake.nix:210-220` - `checks` contains only ruff / mypy / pytest / records.
`nix flake check` builds `checks` and merely evaluates `packages`, so neither
`packages.scufris` (the runtime venv + `mkApplication`) nor `packages.web`
(`buildNpmPackage`) is ever built by CI. The concrete failure: a routine
`web/package-lock.json` change makes `npmDepsHash` (`flake.nix:119`) stale.
The `web` job's `npm ci` passes happily - it does not touch Nix - and the `nix`
job passes because `.#web` is never built. CI is green, and then
`nix build .#web` (and therefore the NixOS/home-manager service modules that
resolve `self.packages.${system}.web`) fails on every consumer. Same class of
gap for `.#scufris`: the checks use the dev `virtualenv`, not `runtimeVenv`.

AGENTS.md's new text asserts "CI enforces that gate, and CI is what decides
green", which makes this gap read as a stronger promise than the workflow
delivers.

Suggested change: either add `nix build .#scufris .#web` as an extra step in the
`nix` job, or (cleaner, keeps one gate) add them to `checks` so a local
`nix flake check` catches it too. If it is deliberately out of scope, say so in
a workflow comment the way the vm-test absence is called out, and open a
follow-up task.

### MINOR `records` check lacks the sandbox hygiene `mkCheck` has

`flake.nix:156-166` - `recordsCheck` copies the tree the same way `mkCheck` does
but omits `export HOME=$TMPDIR` and does not put `pkgs.git` / `pkgs.cacert` in
`nativeBuildInputs`. In the Nix sandbox `HOME` is `/homeless-shelter` (which does
not exist), so the moment `tatr` reads or writes a config/cache/state dir under
`$HOME`, or shells out to `git`, this check fails with an error that has nothing
to do with the task records. It happens to pass today; that is a property of the
current `tatr`, not of this derivation.

Suggested change: factor the shared preamble out - e.g. make `mkCheck` take the
toolchain as an argument (`mkCheckWith = tools: name: command:`) and define
`records = mkCheckWith [inputs'.tatr.packages.default] "records" "tatr check --ledger LESSONS.md"`.
That removes the duplicated `cp -r`/`chmod`/`cd` block at the same time.

### MINOR The pinned-tatr argument does not hold for the command AGENTS.md tells developers to run

`flake.nix:28-36` claims the input exists "so the conformance gate is the SAME
code locally and on the runner, pinned by flake.lock". That is true for
`nix flake check`, but `tatr` is not in `devShells.default.packages`
(`flake.nix:222-231`), while AGENTS.md ("Development flow", "Tasks, tags,
versioning") instructs agents to run `tatr ls`, `tatr new`, and
`tatr check --ledger LESSONS.md` directly. Those run whatever `tatr` is on the
user's PATH - an unpinned, globally installed build. So the version that lints
records interactively and the version that gates CI can silently disagree, which
is the exact drift the comment says has been eliminated.

Suggested change: add `inputs'.tatr.packages.default` to the devShell's
`packages` list, so `nix develop` gives the pinned binary.

### MINOR AGENTS.md claims the workflow's commands are documented in AGENTS.md, but the web commands are not

`AGENTS.md:79-84` (new text): "another runs `cd web && npm run ci` (prettier,
eslint, vitest, webpack build). Both run the SAME commands this file documents -
if you ever need a different command in the workflow than the one here, the gate
has drifted and that is the bug." The "Build, run, test" command block
(`AGENTS.md:56-70`) documents only the Python/Nix commands; `npm ci` and
`npm run ci` appear nowhere in this file except inside that very sentence. A
reader following the stated drift-detection procedure has nothing to compare
against for half the gate.

Suggested change: add the two web commands to the "Build, run, test" block:

```sh
cd web && npm ci            # install frontend deps (lockfile-exact)
cd web && npm run ci        # frontend gate: prettier + eslint + vitest + build
```

### MINOR CHANGELOG.md not updated

Repo AGENTS.md ("Tasks, tags, versioning"): "Notable changes go to
`CHANGELOG.md` (Keep a Changelog)." `CHANGELOG.md` has an `## [Unreleased]` /
`### Added` section that this diff does not touch. Adding the repository's first
CI gate is a notable change for a `v0.1.0`-tagged task.

Suggested change: one bullet under `## [Unreleased]` / `### Added` noting that
every push to master and every pull request now runs `nix flake check` and the
web gate, and that CI is the source of truth for green.

### NIT `cancel-in-progress` also cancels master pushes

`.github/workflows/ci.yaml:14-17` - the concurrency group is `ci-${{ github.ref }}`
with `cancel-in-progress: true`, unconditionally. On a PR that is exactly right.
On `refs/heads/master`, two commits landing close together mean the first one's
run is cancelled and that commit never gets a verdict of its own - which
undercuts the story's "a broken master is noticed by the repository". Given the
observed ~2 min runtime this costs almost nothing to fix.

Suggested change:

```yaml
concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
```

### NIT Node pinned by major in CI, exactly in Nix

`.github/workflows/ci.yaml:59-65` - the comment says "Node 24 matches
`pkgs.nodejs` in the dev shell (node v24.18.0 as of this commit)". I verified
`pkgs.nodejs` is indeed 24.18.0 on the locked nixpkgs, so the claim is accurate
today. But `node-version: "24"` floats across the whole major while the flake
pins an exact patch, so the two drift apart the moment setup-node ships 24.19.
Harmless in practice; if you want the comment's promise to be enforceable rather
than aspirational, pin the exact version, and note that a nixpkgs bump then
requires touching the workflow too (which is arguably the point).

### NIT Prefer the flake-parts `inputs'` accessor

`flake.nix:158` - `inputs.tatr.packages.${system}.default` works, but `perSystem`
already receives `inputs'`, so `inputs'.tatr.packages.default` is the idiomatic
form and drops the manual `${system}` threading. I confirmed `tatr` does define
all four systems in this flake's `systems` list (evaluated
`github:alexjercan/tatr#packages.aarch64-darwin.default` successfully), so there
is no cross-platform correctness problem here - purely a style point.

## What is good

- The net diff is clean: the deliberate break commit (`bdf1e93`, touching
  `scufris/health.py` and `web/src/common.ts`) is fully reverted by `7b24f64`,
  and `git diff master...HEAD -- scufris web tests` is empty. Nothing of the
  break survives.
- No silent skip anywhere. Both jobs run unconditionally on every push to master
  and every PR, with no path filters, no `continue-on-error`, no `|| true`, and
  no pipe that could eat an exit code - the global AGENTS.md shell rule is
  respected. I read the logs of run 30443720539 and confirmed the checks really
  executed (`scufris-ruff> All checks passed!`, `scufris-mypy> Success: no issues
  found in 63 source files`, `scufris-pytest> 567 passed, 15 skipped`,
  `building '...scufris-records.drv'`, and 189 vitest tests plus a webpack
  build), and the logs of run 30443929343 confirming BOTH jobs went red on the
  break. That is real proof, not an assumption.
- Running the whole Python gate as one `nix flake check` rather than three
  split steps is the right call, and the comment explaining why (splitting means
  either re-entering the flake or bypassing it, and bypassing is how gates drift)
  is exactly the kind of WHY comment the task asked for. Same for
  `--print-build-logs`.
- Putting `records` in `checks` rather than as a bare CI step is the right
  instinct: one gate, catchable locally.
- The vm-test exclusion is documented in the workflow header and in AGENTS.md,
  so its absence reads as a decision.
- DECISION.md is genuinely good: three mutually exclusive options, the operator
  cost of each stated up front, and an explicit trigger for re-opening the
  Cachix question.
- All new text is plain ASCII - no em dashes, smart quotes, or arrows - and the
  commits carry no AI attribution.
