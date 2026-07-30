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
| `README.md` | Setup only: what Scufris is, how to run/deploy it, every env var, how to enable each feature. |
| `scufris/README.md` | The architecture: processes, trust boundaries, the approval contract, MCP audiences, module map. |
| `scufris/host/README.md` | The read-only inspection package and the three rules it reads by. |
| `scufris/hostd/README.md` | The root helper: nix options, the socket language, every verb, the audit log. |
| `web/README.md` | The frontend: pages, entries, build, the `npm run ci` gate. |
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
nix develop                       # enter the dev shell (venv already on PATH, activated)
scufris                           # run the app (console entry point)
python -m scufris                 # same, via the module
uv run scufris                    # run through uv inside the shell

ruff check .                      # lint
ruff format .                     # format
mypy .                            # type-check
python -m pytest                  # tests (use `-m`, not bare `pytest`, in a worktree)

nix flake check                   # the full QA gate: ruff + mypy + pytest + records
nix build .#scufris .#scufris-web # build what a release ships (flake check only evaluates these)
nix run .#scufris                 # build and run it

cd web && npm ci                  # install the frontend deps (once per worktree)
cd web && npm run ci              # the frontend gate: prettier + eslint + vitest + build
```

`nix flake check` is the source of truth for green - it runs `ruff check .`,
`mypy .` and `pytest` each against a fresh writable copy of the tree
(`mkCheck` in `flake.nix`), plus a `records` check that runs
`tatr check --ledger LESSONS.md` over the task records. Run the fast local
equivalents while iterating; run at least the checks your change touches
before calling it done, and say plainly when you skipped one.

**CI enforces that gate, and CI is what decides green.**
`.github/workflows/ci.yaml` runs on every push to master and every pull
request: one job runs `nix flake check` and then `nix build .#scufris .#scufris-web`,
another runs `cd web && npm run ci` (prettier, eslint, vitest, webpack build).
The explicit `nix build` matters - `nix flake check` only EVALUATES `packages`,
so without it a stale `npmDepsHash` would sail through green while the flake is
broken for anyone consuming it. Both jobs run the SAME commands this
file documents - if you ever need a different command in the workflow than the
one here, the gate has drifted and that is the bug. A local pass is a good
prediction of green; the run on the pull request is the answer.

The NixOS VM test (`nix build .#scufris-vm-test`) is NOT in CI - it needs KVM, and it
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

## Deployment and authentication

The bind address decides the security posture, and the code enforces it rather
than trusting the operator to remember:

- `SCUFRIS_HOST=127.0.0.1` (the default, and what tests/examples use): no
  authentication, because nothing off the machine can reach it.
- Any other bind: an operator session is REQUIRED, and `create_app` raises
  `AuthConfigError` when no `SCUFRIS_AUTH_PASSWORD_HASH` is configured. The
  deployed unit fails to start rather than serving open.

Generate the hash with `scufris hash-password` and add the printed line to
`sops secrets/scufris.env` in `~/personal/nix.dotfiles` - the same decrypted
dotenv the unit already takes as its `EnvironmentFile` for
`SCUFRIS_TELEGRAM_BOT_TOKEN`. The password never enters the repo, a log, or an
agent transcript.

Enforcement is ONE deny-by-default HTTP middleware in `scufris/app.py`, with a
tiny public allowlist in `scufris/auth.py` (`PUBLIC_PATHS`,
`PUBLIC_STATIC_PATHS`). Do not add per-route auth dependencies: a new route must
be protected by default, and `tests/test_auth.py` enumerates `app.routes` to
prove every non-public one is. Frontend calls go through `apiFetch` in
`web/src/common.ts` for the same reason (CSRF header + 401 redirect in one
place); a vitest guard fails the build on a bare `fetch(` outside that seam.

The app calls its own API from MCP tool subprocesses
(`mcp_common._api_call`). Those authenticate with the per-process
`SCUFRIS_API_TOKEN` bearer token minted in `create_app`, NOT with a cookie and
NOT by trusting loopback. That token is refused outright on the host-action
decision endpoints (`auth.OPERATOR_ONLY_PATTERN`) - see the privileged-actions
section below. A new server registration in
`agent.scufris_mcp_servers` that calls the API must carry that env var.

The rationale, threat model, and the explicitly unsupported deployments are in
`tasks/20260729-125015/DECISION.md`.

## Privileged host actions (scufris-hostd)

Reading the host needs no privilege and lives in `scufris/host/`. CHANGING it
lives in `scufris/hostd/`, runs in a SEPARATE root process, and goes through one
contract with no exceptions:

    propose -> preview -> approve -> apply -> audit -> roll back

- **The mutating tools belong to ONE audience: the host agent.** The MCP audience
  split is physical (`enums.audience_for` decides it, `agent.scufris_mcp_servers`
  wires it): an orchestrator turn gets `scufris` + `den`, the reserved HOST agent
  (`enums.HOST_AGENT_ID`, `/agents/host`) gets `host` + `agent`, a project sub-agent
  gets `agent` only. `mcp_host_tools` defines the host toolset once and registers
  the INSPECTION half on the orchestrator too, so "why is this box hot" stays a
  direct answer; the propose tools exist only on the host agent's server, so the
  propose/preview/approve contract is stated in exactly one steering preamble
  (`sessions.HOST_STEERING_PREAMBLE`) and the orchestrator is steered to DELEGATE a
  change (`run_agent("host", goal)`) rather than reach for a shell that cannot do it.
  See `tasks/20260729-125040/DECISION.md`.
- **A pending approval is a BLOCKED agent, and the operator is the decider.** The
  requesting agent gets an `AgentState.BLOCKED` outcome (not `WAITING`, which means
  "the orchestrator owes it an answer"), so it shows in `pending_agents` as
  something the orchestrator must not answer - and the chat route refuses an
  agent-credential message to it, because "approved, go ahead" is not the
  orchestrator's to say. The decision resumes that agent with the applied result or
  the denial reason (`host_approvals.decision_message`), deferred until any in-flight
  turn ends.
- **One decision path, two surfaces.** `HostApprovalService` (`host_approvals.py`)
  owns approve/deny/cancel/revert; the HTTP routes and the Telegram bot are
  translators that supply an ACTOR string derived from their own credential
  (`app._build_telegram_approval_ops` turns a chat id into
  `operator:telegram:<chat_id>` and re-checks the allowlist, so the transport never
  supplies an actor string of its own). Every rule after that - already decided, expired, drifted, one-way
  acknowledgement, the race - has one implementation. Do not add a second one: an
  allowlisted Telegram chat counts as the operator by decision, and the way that
  stays safe is that it gains no rule of its own. Both surfaces also render from
  ONE renderer (`host_actions.render_action`) and offer a control only where
  `HostApprovalService.decidable()` says a decision can still be made - a queue that
  offers a button the service would refuse is a queue that lies about what the
  operator can do.
- **The strong confirmation is for what DESTROYS something.**
  `host_approvals.confirmation_for` requires an explicit acknowledgement token when
  an action is irreversible AND not mere service control. Keying it on
  `reversal.possible` alone was tried and refuted by measurement: for R1, "no undo"
  is the normal answer (restarting a running unit ends where it started), so that
  rule demanded a typed acknowledgement for every service restart - which is both
  wrong about the risk and self-defeating, since a warning that fires on the routine
  act is why nobody reads the one on `gc_store`.
- `scufris/hostd/` is the helper. It runs as root under
  `services.scufris-hostd` (`nix/scufris-hostd.nix`, exported as
  `nixosModules.scufris-hostd` - separate from the app module ON PURPOSE) and speaks
  typed JSON frames over a unix socket. **The verb set IS the risk taxonomy.**
  R1 is service control (reversible), R2 is disposable cleanup (one-way), R3 is
  the config change (`activate`, `rollback`), and R4 - partitioning, users, key
  material, the firewall, scufris itself - is enforced by NO VERB EXISTING, not
  by a deny check. There is no shell verb at any privilege under any approval;
  do not add one.
- **A plan is a SEQUENCE of steps, and a half-applied one says so.** Activation
  is two commands (point the system profile at the path, then switch to it), so
  `Plan.steps` is a list and the record reports how far it got. Where stopping
  between steps is itself a state - R3's is "this boot runs the old
  configuration, the next boot runs the new one" - the plan carries that sentence
  and the audit repeats it. A multi-step failure must never be logged as
  "nothing happened".
- **Refuse the TYPE, not the name.** R1 acts on services, sockets, timers, paths
  and mounts only. Targets, slices and scopes have no code path, because that is
  how a deny-list of service names gets walked around: `emergency.target` drops
  the box to single-user and kills sshd without ever naming sshd, and
  `user@1000.service` ends the session the scufris USER service lives in.
  Enumerating dangerous names inside an allowed type is a game of catch-up
  (20260729-125029 review round 1, R1.5).
- **The helper builds every argv.** A caller names a verb and typed arguments.
  Adding a capability is a reviewed code change with a test, never a
  configuration line.
- **The helper holds every proposal, and the app rebuilds its queue from it.** The
  app's registry (`host_actions.HostActionStore`) is in-memory by design, so a
  restart inside a proposal's ten-minute window would otherwise strand a live
  approval. The read-only `list_pending` verb is how it recovers - the helper stays
  the single source of truth rather than the app persisting a second copy next to
  the root-owned audit log. It also means the queue shows a proposal made by another
  client of the socket. Only ADDITIONS are applied: an absence cannot be told apart
  from "expired", "denied elsewhere" or "just applied", and the decision path
  refuses an undecidable proposal anyway.
- **The helper holds every proposal.** `apply(id)` is the only way to act, so
  "preview one thing, apply another" has no code path. Apply is guarded by four
  distinct refusals: unknown id, already used, expired, and the system having
  drifted since the preview.
- **Approval is an operator act, and that needs a SESSION.** The decision
  endpoints (`auth.OPERATOR_ONLY_PATTERN`) require a real operator session
  whatever the bind address, checked before the middleware's bearer
  short-circuit AND before the `auth_on` short-circuit, with CSRF and origin in
  the same self-contained block. Asking "is a bearer token present?" is NOT
  enough - a caller sending no header at all then reaches `/approve` on a
  loopback deployment and runs a root command anonymously, which the model's own
  shell can do with `curl` (review round 1, R1.1). `create_app` also refuses to
  build an app that has `hostd_secret` set and no `auth_password_hash`: host
  agency with nobody to approve is not a deployment. There is deliberately no
  approve MCP tool; `tests/test_host_mcp_server.py` asserts the absence, and
  `test_every_mutating_host_route_is_operator_only` enumerates `app.routes` so a
  new host route cannot quietly miss the pattern.
- **"Who asked" comes from the credential, never the body.** A bearer caller is
  an agent; a session is the operator. Reading it from a request field meant
  every agent proposal was audited as the operator (review round 1, R1.6).
- **The audit is the helper's, not the app's.** Root-owned, append-only,
  size-rotated (16 MiB x 5, minimum 2 files so rotation is never deletion), no
  verb deletes an entry. It has to be trustworthy when the app is the thing that
  misbehaved. Refusal records COALESCE per window: reaching the socket needs no
  secret, so a record per connection let an unauthenticated caller rotate the
  whole history off disk in about ninety seconds (review round 1, R1.2). Any new
  audit write on an unauthenticated path needs the same treatment.
- **Two secrets, two files.** `SCUFRIS_HOSTD_SECRET` goes in the same
  `sops secrets/scufris.env` as `SCUFRIS_AUTH_PASSWORD_HASH`, and the same value
  in a file the helper reads via `secretFile`. Unlike the machine API token it
  CANNOT be kept out of `os.environ` - an `EnvironmentFile` is how a sops secret
  reaches the unit - so it is STRIPPED from every subprocess environment instead
  (`config.SECRET_ENV_VARS`, applied in `agent.agent_subprocess_env`, which is
  the ONE place a child environment is built). Without that the agent CLI, and
  every shell command the model runs from it, holds the credential for the root
  socket (review round 1, R1.3). Any new credential on `Settings` must join that
  set, and any new agent spawn must pass that environment; two tests enforce it -
  one enumerates secret-shaped fields, the other walks the package's AST and
  fails a `create_subprocess_*` that inherits the environment unstripped (review
  round 2, R2.1, after the claude backend did exactly that).
- Tests inject a `Runner` (canned command output), an `Executor` (a scripted
  apply) and a `Files` (the store questions R3 asks), so the whole path including
  cancellation runs without root. The half that cannot be faked - a real root
  unit on a real socket, and a REAL activation and rollback of a real second
  toplevel - is `nix build .#scufris-hostd-vm-test`, which is NOT in `nix flake check`
  (it needs KVM) and runs in the release pipeline.
- **Every `nix` (new CLI) invocation goes through `host.run.nix_cli`.** It adds
  `--extra-experimental-features "nix-command flakes"`, because whether those are
  enabled is the operator's `nix.conf` and not something this code controls -
  measured in the VM test, a default configuration makes `nix path-info` fail
  outright. `nix-env` and `nix-store` (the old CLI) need nothing.

## R3: changing the NixOS configuration

The rule that shapes this whole feature: **the configuration repository is a
PROJECT, and Scufris does not edit it.** An agent changes `~/personal/nix.dotfiles`
the way it changes any project - a sprout worktree, a commit on a branch, a
review. There is no configuration editor here, no typed "add a package" verb, and
no code path that writes to that repository. `tasks/20260729-125035/DECISION.md`
records why (and why typed edit verbs were rejected).

What Scufris owns is the last mile, in `scufris/hostconfig.py` (unprivileged) and
`scufris/hostd/nixos.py` (the previews):

- **Build from a commit, as the operator.** A ref is resolved to a rev and built
  as `git+file://<repo>?ref=&rev=#nixosConfigurations.<attr>...toplevel`. Never
  as root: nix EVALUATION reads files as the evaluating user, so a configuration
  evaluated as root could read a host key or a sops age key into a derivation
  output. Building from `?rev=` also means the tree comes from the commit, so
  uncommitted files are structurally excluded (and reported), and the flow cannot
  dirty the repository.
- **A caller may not supply a toplevel.** `POST /api/host/actions` and
  `propose_host_action` refuse `kind=activate` outright, and the helper validates
  the path anyway (a store-path ROOT, valid in this store, carrying
  `nixos-version` and `bin/switch-to-configuration`). A caller who chose the
  store path would be choosing what the machine boots while the closure diff
  described their choice faithfully.
- **The preview does not run the proposed configuration.** The unit-restart list
  can only come from that configuration's own `switch-to-configuration`, as root,
  before anyone approved it - so it is not shown, and the preview says why. The
  diff is `nix store diff-closures`, whose measured trap is handled: identical
  closures print NOTHING on exit 0, so "no closure change" is stated explicitly.
  Its ANSI codes and non-ASCII glyphs are stripped at the source.
- **A switch that is already running blocks the next one.** The apply-time
  preflight refuses when `nixos-rebuild-switch-to-configuration.service` is
  active - the same transient unit name `nixos-rebuild` uses, so a hand-run
  switch and this helper cannot interleave - and refuses when it cannot tell.
- **Rollback names a NUMBER.** The helper resolves that generation's store path
  from the profile; an applied activation records the generation it replaced and
  offers exactly that rollback. `nixos-rebuild --rollback` ("whatever is
  previous") is deliberately not used.
- The residual risk, stated rather than engineered away: an activated
  configuration can run anything as root. The controls are the reviewed commit,
  the diff the operator reads, and the audit record naming the revision.

The decision record is `tasks/20260729-125020/DECISION.md`; the three forks the
framework task settled are `tasks/20260729-125029/DECISION.md`, and the fork R3
turned on is `tasks/20260729-125035/DECISION.md`.

## The scheduled checks (the one proactive surface)

`scheduler.py` owns the clock, `checks.py` owns the judgement, `digest.py` owns the
words, and `app.py` wires them to Telegram and `/host/`. The decision record is
`tasks/20260729-125046/DECISION.md`; four things about it are load-bearing:

- **Two schedules with fixed identities**, `watch` (interval) and `daily` (time of
  day). `watch` delivers only on a warn/crit or a recovery; `daily` always delivers.
  Silence therefore means "nothing is wrong", because the daily line is the
  heartbeat. Config is plain settings fields rather than a schedule language.
- **Nothing fires on a fresh schedule, and a missed window is recorded, not
  replayed.** First sight arms a schedule one window ahead; a window that passed
  while the app was down is counted as missed and skipped. Both matter: the first
  made every app start (including every test that boots one) perform real subprocess
  reads, and the second is what stops an app that was down for six hours from
  delivering twenty-four digests at once.
- **A check never decides what "too full" means** - it reads the threshold from
  settings, all of which are runtime-editable. UNAVAILABLE is not OK, and a check
  that raises or times out becomes a NAMED failure in the digest rather than a
  shorter digest, which would read as good news.
- **Escalation is `checks.ESCALATABLE` and nothing else.** A threshold may propose a
  store collection (R2) and may never propose a unit restart or an activation;
  `escalation_for` raises on anything outside the allowlist, and the proposal goes
  through `HostApprovalService` like any other. Default off.

Reading the host for a check pass is injectable (`create_app(host_inspector=...)`)
for the same reason the NixOS build is: a real pass walks the nix store, so tests
inject an inspector over a `FakeRunner` replaying captured output.

## Docs sync

When a code change makes something in a README, an example, or `.env.example`
wrong, fix it in the SAME task as the code change. A ticked docs step is not
proof - check the surface against the diff.

The live doc surfaces are the root `README.md` (setup, and the full env-var
table), `scufris/README.md` (architecture), `scufris/host/README.md`,
`scufris/hostd/README.md` (the verbs and the wire format), `web/README.md`, this
file, and `.env.example`. A new setting means a row in the root README AND in
`.env.example`; a new verb or frame means `scufris/hostd/README.md`. Task records
under `tasks/` are append-only history and are NOT a doc surface: never rewrite
one to match a later rename.
