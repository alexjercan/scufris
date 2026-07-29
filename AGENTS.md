# AGENTS.md

Orientation for agents working on **Scufris** ("Scuffed Jarvis"), a host
monitoring dashboard with an embedded assistant. Read this first, then check
the backlog (`tatr ls --sort priority`) and the spike docs under `tasks/`
before diving in.

## What this is

Scufris is a single-host dashboard: it shows live stats about the machine it
runs on (CPU, memory, disk, network, processes, temps, ...), lets you drive a
set of local CLI tools through UI elements instead of retyping commands, and
gives you a chat panel to talk to an LLM-backed agent about the box. Think a
scuffed, self-hosted Jarvis for one NixOS machine.

Three pillars, each still being scoped by a spike:

- **Monitoring** - collect and surface host metrics in a dashboard.
- **CLI control** - expose selected local tools as UI actions.
- **Agent** - a chat assistant backed by an external LLM provider
  (target: GPT-5.5, reached through a Pro/Plus subscription, NOT an API key).

The concrete shape of each pillar - dashboard framework, metrics source, and
which LLM harness - is an open question captured as a `/spike` under `tasks/`.
Do not assume; read the spike doc.

## Layout

Python project, packaged with `uv` and built reproducibly with `uv2nix`
through the flake.

| Path | What it is |
|------|------------|
| `scufris/` | The Python package. `__main__.py` is the entry point (`scufris` console script). |
| `flake.nix` | Nix dev shell + `uv2nix` package/venv wiring; `nix flake check` runs the QA gate. |
| `pyproject.toml` | Project metadata, deps, and tool config (ruff, mypy, pytest). |
| `uv.lock` | Locked dependency set; the source of truth uv2nix reads. |
| `tests/` | Pytest suite (harness/integration first - see Testing). |
| `examples/` | Small runnable scripts that exercise a component end to end. |
| `LESSONS.md` | The lessons ledger - read it before starting any task (see below). |
| `tasks/` | tatr task records - one folder per task (TASK/SPIKE/REVIEW/RETRO/NOTES). |
| `CHANGELOG.md` | Keep a Changelog; notable changes land here. |

Current stack (from `pyproject.toml`): FastAPI + Uvicorn (HTTP surface),
httpx/requests (outbound), pydantic + pydantic-settings (models/config),
python-dotenv (`.env`), python-ulid (ids), rich (terminal output). Dev tools:
ruff, mypy, pytest (+ pytest-asyncio, respx). These are a starting point; a
spike may add or swap a dependency (e.g. a TUI or metrics library).

## Build, run, test

The dev shell is provided by the flake and uses a `uv2nix`-built editable
virtualenv. On NixOS:

```sh
nix develop                 # enter the dev shell (venv already on PATH, activated)
scufris                     # run the app (console entry point)
python -m scufris           # same, via the module
uv run scufris              # run through uv inside the shell

ruff check .                # lint
ruff format .               # format
mypy .                      # type-check
python -m pytest            # tests (use `-m`, not bare `pytest`, in a worktree)

nix flake check             # the full QA gate: ruff + mypy + pytest + records
nix build .#scufris .#web   # build what a release ships (flake check only evaluates these)
nix run .#scufris           # build and run it

cd web && npm ci            # install the frontend deps (once per worktree)
cd web && npm run ci        # the frontend gate: prettier + eslint + vitest + build
```

`nix flake check` is the source of truth for green - it runs `ruff check .`,
`mypy .` and `pytest` each against a fresh writable copy of the tree
(`mkCheck` in `flake.nix`), plus a `records` check that runs
`tatr check --ledger LESSONS.md` over the task records. Run the fast local
equivalents while iterating; run at least the checks your change touches
before calling it done, and say plainly when you skipped one.

**CI enforces that gate, and CI is what decides green.**
`.github/workflows/ci.yaml` runs on every push to master and every pull
request: one job runs `nix flake check` and then `nix build .#scufris .#web`,
another runs `cd web && npm run ci` (prettier, eslint, vitest, webpack build).
The explicit `nix build` matters - `nix flake check` only EVALUATES `packages`,
so without it a stale `npmDepsHash` would sail through green while the flake is
broken for anyone consuming it. Both jobs run the SAME commands this
file documents - if you ever need a different command in the workflow than the
one here, the gate has drifted and that is the bug. A local pass is a good
prediction of green; the run on the pull request is the answer.

The NixOS VM test (`nix build .#vm-test`) is NOT in CI - it needs KVM, and it
guards the release pipeline instead (`tasks/20260729-125101/DECISION.md`).

Dependency changes go through uv, then the lock and the flake follow:

```sh
uv add <pkg>                # or edit pyproject.toml + uv lock
uv lock                     # regenerate uv.lock (uv2nix reads this)
```

`UV_NO_SYNC=1` and `UV_PYTHON_DOWNLOADS=never` are set in the shell: the venv
is managed by uv2nix, not uv, and Python comes from Nix. After changing deps,
re-enter `nix develop` (or rebuild) so the venv picks them up.

## Testing: harness-first

Per the global `~/AGENTS.md`, prefer integration and end-to-end tests over
isolated unit tests, and ship a small runnable example for a substantial
component.

- **Bugs: reproduce first.** The first artifact of a bug task is a failing
  test that replays the situation. Then fix; the same test becomes the
  regression pin.
- **Features ship with a test that exercises them the way the app uses them** -
  hit the FastAPI route with a client, drive the collector against a real (or
  faithfully faked) source, run the agent against a stubbed provider. Use
  `respx` to fake HTTP and `pytest-asyncio` for async paths rather than
  mocking the unit under test into meaninglessness.
- An `examples/` script that boots the piece end to end is the cheapest proof
  it works, and doubles as documentation. Add one when it is cheap.
- **Run `python -m pytest`, not bare `pytest`, in a sprout worktree.** The
  console-script `pytest` does not put CWD first on `sys.path`, so it can import
  `scufris` from the main checkout and silently test the wrong tree. `tests/conftest.py`
  guards this: it fails fast with a pointer to `python -m pytest` if `scufris`
  resolves from outside the current directory.

## Conventions

- Global rules from `~/AGENTS.md` apply: plain ASCII punctuation only (`-`,
  `--`, `...`, `->`, straight quotes - no em dashes, smart quotes, or arrows),
  plain commit messages with NO AI attribution, no time-based technical
  arguments, and the shell/verification rules (never let a pipe or echo eat a
  command's exit code; kill helpers by recorded PID, never `pkill -f`).
- Python style is enforced by ruff (line length 88, double quotes, target
  py313) and mypy - keep both clean rather than sprinkling `# type: ignore`.
  Type new code; missing third-party stubs are handled per-module in
  `pyproject.toml`'s mypy overrides, not with blanket ignores in source.
- Config comes through pydantic-settings and `.env` (see `.env.example`);
  never hardcode secrets or read raw `os.environ` scattered around.
- Prefer async where the framework is async (FastAPI/httpx); do not block the
  event loop with sync I/O in request paths.

## Worktrees and shared checkout

- Isolated work happens in a **sprout** worktree (the mechanism `/work` and
  `/flow` use). Never hand-create a worktree and never use
  `.claude/worktrees/`; only ever `cd "$(sprout new <type>/<slug>)"`.
- The main checkout may be shared with parallel sessions. Before every commit
  there, confirm `git branch --show-current` is what you expect, stage
  explicit paths (never `git add -A` in the shared checkout), and glance at
  `git status` so a generated file like `uv.lock` is not dropped. Inside a
  sprout worktree, `git add -A` is fine.
- Commit only when the user asks (global rule). Do not add a Claude co-author
  trailer.
- A versioned pre-commit hook (`hooks/pre-commit`, activated by
  `core.hooksPath=hooks` which the devShell sets on entry) refuses a staged
  `web/node_modules`: in a sprout that path is a symlink to the main checkout's
  node_modules, and `.gitignore`'s `node_modules/` (dir-only) does not match it,
  so `git add -A` would otherwise commit it and corrupt the branch. If you have
  not entered `nix develop`, enable it once with `git config core.hooksPath hooks`.

## Development flow

`/flow` drives development here: it plans a goal into tatr tasks, then runs
`/work` (implement in a sprout worktree), `/review` (out-of-context round-1
review until APPROVE), and `/compound` (retro + lesson) for each one. Task
Definitions of Done carry checkable proofs in `test:` / `cmd:` / `manual:`
notation. `LESSONS.md` at the repo root is the lessons ledger - read it before
starting any task. `tatr check` (plus `tatr check --ledger LESSONS.md`) is the
conformance gate for task records and the ledger; keep it clean.

## Where records go (/plan, /spike, /work, /review, /compound, /flow)

Everything tied to one task lives in that task's folder under `tasks/<id>/` -
never as loose `.md` files under `docs/`:

- `tasks/<id>/TASK.md` - the task (tatr; body shape: Story / Steps /
  Definition of Done / Notes).
- `tasks/<id>/SPIKE.md` - the spike/research doc (`/spike`).
- `tasks/<id>/REVIEW.md` - review rounds and verdict (`/review`).
- `tasks/<id>/RETRO.md` - the retrospective (`/compound`).
- `tasks/<id>/NOTES.md` - design/fix record for the shipped change.

The lessons ledger lives at the repo root as `LESSONS.md` (the ledger
`/compound` appends to, one or two lines per lesson). `docs/` exists only if
there is long-form durable material (design or release plans) to hold; it
currently has none. A spike's SPIKE.md is durable and shared - several tasks
and several `/flow` runs can all cite the same research.

Loose working notes that belong to no single task go in `docs/scratch/`, and
ONLY there. That directory is the ephemeral drawer `/lessons` compiles into
`LESSONS.md` and then empties; `scripts/check-release-ready.sh` refuses to
release while anything is left in it. Durable material elsewhere under `docs/`
is not scratch and does not block a release.

The full lifecycle: `/spike` explores a fuzzy question, `/plan` scopes a
defined feature into steps, sprout isolates, `/work` implements with tests,
`/review` critiques until APPROVE, `/compound` distills the lesson, and
`/flow` drives the whole loop end to end.

## Tasks, tags, versioning

- Tasks: the `tatr` CLI, markdown under `tasks/`. Check the backlog before
  starting (`tatr ls --sort priority`), close tasks when done, and record what
  changed / why / what was hard in the task body so a cold session can follow.
- **Every new tatr task carries exactly one scheduling tag:** `backlog` with
  priority 0 (not yet scheduled), OR the current release tag once a release
  plan exists (e.g. `v0.1.0`) with a priority slotted RELATIVE to that
  release's other open tasks (`tatr ls -f ':tags contains vX.Y.Z' --sort
  priority` first). Until the first release plan is written, `backlog` is the
  default. Topical tags (`spike`, `dashboard`, `agent`, `bug`, `feature`,
  `docs`, `ui`, ...) come on top. Pulling a backlog task into a release = swap
  the tag, re-slot the priority.
- Run one `tatr new` per shell call; on a same-second ID collision it fails
  loudly - retry once the second ticks. Prefer `tatr new -b <file>` (or `-`
  for stdin) to seed the body at creation.
- Version lives in `pyproject.toml`. Notable changes go to `CHANGELOG.md`
  (Keep a Changelog).

## Releasing

`pyproject.toml` holds the version. `CHANGELOG.md` says what is in it. The tag
names it. All three must agree, and `scripts/check-release-ready.sh` is what
proves they do - run it before you tag, because the release pipeline runs the
same script as its guard and will stop you there otherwise.

**When to bump.** Semantic versioning. A release is a deliberate act, not a
consequence of merging: master is always green (CI enforces that), and a
version exists when you decide to name one.

**Cutting a release.** Do all of this **from the MAIN checkout on master, inside
`nix develop`** - not from a sprout worktree. The tag must name a commit that is
on master and already pushed; tagging a feature branch would publish a release
for a commit master does not contain. The dev shell matters because the guard
needs `tatr` on PATH.

```sh
cd ~/personal/scufris && nix develop
git branch --show-current         # must print: master
git pull --ff-only

# 1. Bump the version if it is not already what you intend to ship.
$EDITOR pyproject.toml            # version = "X.Y.Z"

# 2. Move [Unreleased] into a dated section and open a fresh [Unreleased].
#    Refuses if [Unreleased] is empty - there must be something to release.
scripts/cut-changelog.sh X.Y.Z    # idempotent: re-running never moves the date
scripts/cut-changelog.sh --check X.Y.Z

# 3. Read what you are about to publish. This IS the release page.
scripts/release-notes.sh X.Y.Z

# 4. Commit, then prove the tree is releasable. The guard requires a CLEAN
#    tree, so it runs after the commit; if it fails, fix and `git commit
#    --amend` rather than stacking a second commit.
git commit -am "chore: release X.Y.Z"
scripts/check-release-ready.sh vX.Y.Z

# 5. Push master FIRST, and only tag once that push succeeded. Tagging before
#    the push risks a rejected push (someone else moved master) leaving a tag
#    that names a commit the remote does not have.
git push origin master

# 6. Tag the commit you just pushed, and push the tag. The tag is the trigger.
git tag vX.Y.Z
git push origin vX.Y.Z

# 7. Watch THIS release, by tag - a bare `--limit 1` can hand you an older or
#    unrelated run (a manual dispatch, a previous version). --exit-status so a
#    failed run makes this command fail too, instead of exiting 0 on red.
#    Run straight after the push and the run may not be registered yet; that
#    shows up as "failed to get run: HTTP 404" - wait a few seconds and retry.
#    NOTE: --branch matches the TAG only for tag-triggered runs. A re-release
#    started with `gh workflow run` has master as its head branch, so find that
#    one with `gh run list --workflow release.yaml --event workflow_dispatch`.
gh run list --workflow release.yaml --branch vX.Y.Z
gh run watch --exit-status "$(gh run list --workflow release.yaml --branch vX.Y.Z \
    --limit 1 --json databaseId --jq '.[0].databaseId')"
gh release view vX.Y.Z
```

A tag with anything after `MAJOR.MINOR.PATCH` (`v0.2.0rc1`, `v1.0.0.dev4`) is
classified as a pre-release by PEP 440 rules and the release page is marked as
one; `v1.0.0.post1` is NOT a pre-release. The changelog section must be named
for the same version, suffix included.

**What the guard checks** (`scripts/check-release-ready.sh`): the tag,
`pyproject.toml` and the changelog's top released section name the same version;
that version has a dated, non-empty changelog section; `tatr check --ledger
LESSONS.md` is clean; `docs/scratch/` holds nothing uncompiled; the working tree
is clean. Each check prints what it verified and exits non-zero on the first
failure.

**What the pipeline does** (`.github/workflows/release.yaml`): guard, then the
full gate re-run on the tagged commit (including the NixOS VM test, which needs
KVM and so runs nowhere else), then build the wheel and sdist, install the wheel
into a clean virtualenv and check `scufris --version` reports the tagged
version, and only then publish. The release is created as a DRAFT, filled, and
made visible in the final step.

**If the pipeline fails halfway.** No partial RELEASE PAGE is ever visible: a
failure before the last step leaves an unpublished draft, which watchers are not
notified about. The TAG, however, is public the moment you push it, so a failed
release is still a tag people can fetch - that is what the fix-forward advice
below is about.

Re-run the same tag; the publish job is idempotent and updates the existing
release rather than creating a second one:

```sh
gh run rerun "$(gh run list --workflow release.yaml --branch vX.Y.Z \
    --limit 1 --json databaseId --jq '.[0].databaseId')" --failed
```

If the failure is in the repository rather than the runner, the fix needs a new
commit, so the tag has to move. Untag, fix, re-tag - LOCAL delete included, or
the re-tag fails with "already exists":

```sh
git push --delete origin vX.Y.Z    # remove the remote tag
git tag -d vX.Y.Z                  # and the local one
# Only if a draft was actually created - this errors when there is no release:
gh release delete vX.Y.Z --yes
# fix, commit, push master, then tag again as above
```

Moving a tag is only acceptable while the release was never published. Once it
is public, fix forward with a new version instead.

**Yanking a bad release.** These are alternatives, not a sequence - read before
running any of them.

- *Preferred: fix forward.* Ship `vX.Y.Z+1`, and mark the bad version in
  `CHANGELOG.md` with Keep a Changelog's `[YANKED]` marker. Nothing that anyone
  already fetched breaks.
- *Demote it out of "latest":* `gh release edit vX.Y.Z --prerelease=true`. Note
  that this is NOT durable on its own - re-running the release workflow for that
  tag sets `--prerelease` back from the version's own classification. Only use
  it as a stopgap while you prepare the real fix.
- *Remove the page:* `gh release delete vX.Y.Z --yes`. The tag survives, so a
  flake input pinned to it still resolves.
- *Remove the tag too:* `git push --delete origin vX.Y.Z`. Do this ONLY if you
  are confident nobody fetched it. An existing `flake.lock` keeps working, and
  so does a fresh clone of a consumer that committed its lock, because the lock
  records the commit rev rather than the tag. What breaks is re-resolving the
  input: `nix flake update`, or anyone adding the input for the first time.

**Consuming a release.** `~/personal/nix.dotfiles` takes Scufris as a flake
input; pin it to a tag rather than tracking master:

```nix
scufris.url = "github:alexjercan/scufris/vX.Y.Z";
```

## Docs sync

When a code change makes something in the README, an example, or a
`docs/` reference wrong, fix it in the SAME task as the code change. A ticked
docs step is not proof - check the surface against the diff.
