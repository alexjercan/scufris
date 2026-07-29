# Review: Publish a GitHub Release from a version tag

- DATE: 20260729-142231
- ROUND: 2
- REVIEWER: out-of-context agent
- VERDICT: APPROVE

## Findings

Round 1 (commit f71986c). All twelve are addressed in 0bdf18c; see `## Round 2`
at the foot for what was verified and what remains.

### BLOCKER the workflow_dispatch path never checks out the tag, and can create one

`.github/workflows/release.yaml:14-19` accepts a free-form `version` input
described as "must already be tagged", but nothing in the workflow ever
resolves that tag:

- `guard` checks out with no `ref` (line 88-89 for `verify`, 42-47 for `guard`,
  151-152 for `publish`), so on a `workflow_dispatch` every job checks out
  `github.ref` - the BRANCH the run was dispatched from, normally `master`.
- Nothing asserts `refs/tags/$TAG` exists or points at `HEAD`.
- `gh release create "$TAG"` (line 198) has no `--verify-tag` and no
  `--target`. When the tag does not exist, `gh` CREATES it, at the default
  branch head.

So dispatching `v0.1.0` from master on a day master has moved past the tag
does all of: run the "full gate on the TAGGED commit" against master, build and
smoke-test a wheel from master, and publish those artifacts under a tag that
either points somewhere else entirely or gets invented on the spot. The guard
cannot catch it: it compares the tag string to `pyproject.toml` in the checked
out tree, and master's `pyproject.toml` still says `0.1.0`, so it passes. This
is the "publishes something wrong under a permanent version number" failure,
reachable by the exact button the workflow puts on the UI, and the NOTES.md
plan ("dispatch with a deliberately WRONG version as a safe negative proof")
does not exercise it because a wrong version stops at the guard.

Concrete change: add `ref: ${{ needs.guard.outputs.tag }}` to the `verify` and
`publish` checkouts (and check out the tag in `guard` too - it needs
`fetch-tags: true` or a `git fetch --tags` first, since the resolve step runs
before it), add an explicit `git rev-parse --verify "refs/tags/$TAG^{commit}"`
assertion in the guard so a dispatch for an untagged version fails loudly, and
pass `--verify-tag` to `gh release create` so the publish step can never mint a
tag as a side effect.

### MAJOR untrusted input is interpolated into shell in two places

Two script-injection sinks:

1. `.github/workflows/release.yaml:55` -
   `tag="${{ inputs.version || github.ref_name }}"`. This is GitHub template
   expansion, spliced into the `run` script text before bash ever sees it, so
   no amount of quoting in the YAML helps. `workflow_dispatch` inputs are free
   form: a dispatch with version `v0.1.0"; curl ... | sh; "` executes in the
   guard job. The push path is only slightly narrower: the tag filter is
   `v[0-9]+.[0-9]+.[0-9]+*` and that trailing `*` matches anything, `;` and `"`
   included.
2. `.github/workflows/release.yaml:166-172` - `"$VERSION"` is spliced into the
   nested `nix develop --command bash -euo pipefail -c '...'` by closing and
   reopening the outer single quotes, which lands the value in UNQUOTED inner
   context on line 168 and inside inner double quotes on line 171. I reproduced
   both: `VERSION='0.1.0; touch /tmp/X; :'` executes `touch` on line 168, and
   `VERSION='0.1.0$(touch /tmp/X)'` executes it on line 171. The nesting only
   looks safe because the outer `"$VERSION"` is quoted - that quoting protects
   the OUTER shell, not the inner one.

Yes, both require repository write access today, and the guard's version
agreement blocks (2) unless the attacker also lands a `pyproject.toml` change.
That is a mitigation, not a design: this is release automation for a public
repo and it is the one place a hardening habit is cheap. Concrete change: in
the resolve step take the value through the environment
(`env: { RAW_TAG: ${{ inputs.version || github.ref_name }} }` then
`tag="$RAW_TAG"`), validate it against `^v[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9.+-]*$`
and reject anything else before it is used; and in the smoke step stop splicing
- keep the inner script fully single-quoted and let `VERSION` reach it through
the environment, which `nix develop --command` already passes through.

### MAJOR a failed asset upload leaves a live release with nothing on it

`.github/workflows/release.yaml:184-206` creates the release PUBLISHED
(line 198, no `--draft`) and only then uploads (line 203). If `gh release
upload` fails - a flaky upload is the single most likely failure in this job -
what is left is a published GitHub Release, visible, with notes, watchers
notified, and no artifacts. That is precisely the half-created release TASK.md
Step 6 says must not happen ("a failed publish leaves no half-created
release"), the step is ticked `[x]`, and NOTES.md's "Idempotence" section
claims the sequence avoids "leaving a half-created one". A re-run does converge;
the first run's window is the problem, and that window is when a human is
watching a release page appear.

Concrete change: `gh release create "$TAG" --draft ...`, then upload, then
`gh release edit "$TAG" --draft=false --latest=...`. Publication becomes the
last, cheapest, most atomic step, and a failure anywhere before it leaves only
a draft nobody sees.

### MAJOR the docs/ scratch check will block releases on material AGENTS.md sanctions

`scripts/check-release-ready.sh:62-71` fails the release if `docs/` contains
any file not named `README.md`. But AGENTS.md says `docs/` "exists only if
there is long-form durable material (design or release plans) to hold" - i.e.
a `docs/release-plan.md` is EXPECTED and correct. The first time someone writes
one, every release stops with "docs/ holds uncompiled scratch - run /lessons",
which is wrong and unactionable (running `/lessons` will not remove a durable
design doc). The check is a guess at a convention that does not exist in this
repo: there is no defined `docs/` scratch location, and today `docs/` does not
exist at all, so the check is entirely untested against a true positive.

Concrete change: ground the check in a real marker rather than "any file in
docs/" - e.g. fail only on an explicit scratch path (`docs/scratch/**`, or
whatever `/lessons` actually leaves behind), or drop the check and say in
NOTES.md that the epic's DoD-4 scratch clause is carried by `tatr check
--ledger LESSONS.md`. A guard that cries wolf at 2am gets deleted, taking the
real checks with it.

### MAJOR `find | head` inside a command substitution can kill the guard with a bare exit 141

`scripts/check-release-ready.sh:63`:
`stray="$(find docs -type f ! -name README.md | head -20)"`, under
`set -euo pipefail` (line 13). When `find`'s output exceeds the pipe buffer,
`head` exits after 20 lines, `find` dies of SIGPIPE, `pipefail` propagates 141,
and `set -e` aborts the script AT THE ASSIGNMENT - before the `echo "$stray"`
and the `fail` message. Reproduced: with 4000 files under `docs/` the script
exits 141 having printed nothing about why. This is the repo's own documented
rule ("never end a command with a pipe that eats its exit code", AGENTS.md;
LESSONS.md:120 and :184 both record instances). The failure direction is safe
but the diagnostic - the entire point of the check - is what gets eaten.

Concrete change: `stray="$(find docs -type f ! -name README.md | head -20 || true)"`,
or better, capture the full list and truncate in the printing:
`mapfile -t stray < <(find docs -type f ! -name README.md)`.

### MINOR `gh release edit` can never clear a pre-release marking

`.github/workflows/release.yaml:186-195`: `prerelease_flag` is either
`--prerelease` or empty. On the update path, a release previously published as
a pre-release stays a pre-release forever, because the false case passes no
flag at all rather than passing `--prerelease=false`. That is exactly the
re-run scenario the idempotence step is about: fix `v0.2.0rc1`, retag as
`v0.2.0`... except the tag differs, so the realistic case is a release created
before a `pyproject.toml` correction. Concrete change: drop the variable and
pass `--prerelease=$PRERELEASE` to both `create` and `edit` (gh accepts the
`=value` form), which also removes the unquoted-expansion word-splitting on
lines 195 and 201.

### MINOR non-canonical PEP 440 versions break publish and are misclassified

Three related edges around `.github/workflows/release.yaml:63-67` and
`168`/`204-205`:

- The pre-release regex treats "anything that is not `N.N.N`" as a
  pre-release. `0.2.0rc1` and `1.0.0.dev4` are correctly flagged, but
  `0.1.0.post1` (a POST-release) and `0.1.0+local` are flagged as
  pre-releases too. The comment on line 60 says "rc1, b2, .dev4", so the
  intent is narrower than the implementation.
- `dist/scufris-${VERSION}-py3-none-any.whl` and `dist/scufris-${VERSION}.tar.gz`
  assume `VERSION` is already in PEP 440/PEP 625 canonical form. A
  `pyproject.toml` version of `0.2.0-rc1` builds as `scufris-0.2.0rc1-...whl`
  and the publish step fails on a missing file.
- `test "$reported" = "scufris $VERSION"` compares against
  `importlib.metadata`, which reports the NORMALIZED version, so the same case
  fails the smoke test with a confusing message.

Concrete change: glob the artifacts (`dist/*.whl`, `dist/*.tar.gz`, asserting
exactly one of each) rather than reconstructing their names, and compare the
smoke output against the normalized version.

### MINOR the clean-tree check is vacuous in CI, and the check that would matter there is absent

`scripts/check-release-ready.sh:75-79` is a good local check and a no-op in the
guard job: `actions/checkout` always produces a clean tree. The invariant that
actually needs asserting on the runner is the one the local check stands in for
- "the artifacts are built from the commit the tag names" - and nothing asserts
it (see the BLOCKER). Worth a one-line comment saying the check is for the
operator's local run, plus the `git rev-parse` tag assertion in the workflow.

### MINOR `fetch-depth: 0` is justified by a reason that would break the verify job

`.github/workflows/release.yaml:44-47` says the full history is needed because
`tatr check` "needs the whole tree, not a shallow slice of it". `verify` runs
`nix flake check`, which includes the `records` check - the SAME
`tatr check --ledger LESSONS.md` (flake.nix:217) - on a default shallow
checkout (line 88-89). Either the comment is wrong (it is: `tatr check` reads
files, not history) or `verify` is broken. Fix the comment, or keep
`fetch-depth: 0` for the tag-resolution work the BLOCKER fix needs and say so.

### MINOR concurrency does not serialize a dispatch against a tag push

`.github/workflows/release.yaml:29`: `group: release-${{ github.ref }}`. For a
tag push that is `refs/tags/v0.1.0`; for a dispatch of the same version it is
`refs/heads/master`. Two runs publishing the SAME release can therefore run
concurrently, which is the collision the group exists to prevent. Key it on the
version instead: `release-${{ inputs.version || github.ref_name }}`.

### MINOR TASK.md ticks a step the code does not satisfy

`tasks/20260729-125101/TASK.md:36-38` marks "a failed publish leaves no
half-created release" `[x]`, and NOTES.md's "Idempotence" section asserts the
same. Neither is true today (see the MAJOR above). The rest of the record is
notably honest - "What is NOT proven yet" names the two outstanding DoD items
and the plan to close them - which makes this one overstated tick stand out
rather than blend in. Either implement the draft flow or untick the step and
record the gap next to the other two.

### NIT no changelog entry for the release pipeline

`CHANGELOG.md`'s `[0.1.0]` section has an entry for the CI workflow added by
the sibling task 20260729-125051, but nothing for the release workflow or
`scripts/check-release-ready.sh` - and `[Unreleased]` is empty. Since `v0.1.0`
has not been tagged yet, this belongs in the `[0.1.0]` section alongside the CI
entry; otherwise the release page for 0.1.0 will not mention the machinery that
produced it.

### NIT run blocks do not set `pipefail`

The GitHub default shell is `bash -e {0}`, so `-u` and `-o pipefail` are off in
every `run` block except the nested `bash -euo pipefail -c`. Nothing currently
pipes in those blocks, so this is latent, but AGENTS.md's shell rule is
explicit and a workflow-level `defaults.run.shell: bash` plus `set -euo
pipefail` at the top of the multi-line steps costs nothing.

## What is good

The three-job chain is the right shape: cheap guard first, everything
`needs` it, and `contents: write` scoped to the single job that writes. Actions
are SHA-pinned with version comments, matching ci.yaml. `check-release-ready.sh`
is genuinely well built - it delegates parsing to `release_tools.py` instead of
re-implementing it, every branch prints what it verified, `|| fail` is used
correctly so `set -e` does not swallow the diagnostics (except at line 63), and
the missing-`tatr` case fails loudly rather than skipping. I ran it: `v0.1.0`
passes all five checks, `v9.9.9` fails with the exact message NOTES.md claims.

The KVM handling is the best thing in the branch. The DECISION said "attempt it,
remove it if the runner cannot", the probe found a third answer the plan had not
imagined - present but unusable - and the workflow encodes the RIGHT lesson from
it: no `if [ -e /dev/kvm ]`, fix the permission and let the step go red if KVM
ever disappears. The reasoning in DECISION.md and NOTES.md matches what
release.yaml:128-135 actually does, run number and timing included. The
"What is NOT proven yet" section is honest about the two DoD items the branch
cannot close before a tag exists, and names who closes them.

## Round 2

- DATE: 20260729-144512
- COMMIT: 0bdf18c
- VERDICT: APPROVE

All twelve round-1 findings are addressed, and the three I was asked to attack
again hold up under a second attempt. Two minors and two nits remain; none of
them blocks the release, and none is a repeat of a round-1 finding.

### Verified fixed

**BLOCKER (dispatch never checked out the tag).** Fixed properly, and by more
than one mechanism. The resolve step now runs BEFORE any checkout so it can
feed `ref:`, all three jobs check out `ref: <tag>`, and `gh release create`
carries `--verify-tag` so the publish step can no longer invent a tag as a side
effect. `--draft` closes it a third time: GitHub never creates a tag ref for a
draft release at all. The `git rev-list -n 1 "$TAG"` vs `git rev-parse HEAD`
assertion in `verify` (release.yaml:116-125) is a real check rather than a
tautology, and for a reason worth naming: `actions/checkout` resolves a bare
`ref:` as a BRANCH first and only then as a tag, while `git rev-list` prefers
the tag - so a branch that shares a name with the release tag is caught here
and nowhere else.

**MAJOR (injection).** Both sinks are closed. `inputs.version || github.ref_name`
now reaches the shell only through `env: RAW_TAG` (release.yaml:50-51), which
a template expansion cannot escape. The smoke step no longer splices: the inner
script is one unbroken single-quoted string and `$VERSION` is expanded by the
INNER bash from the environment. I re-ran my round-1 payloads against the new
form - `VERSION='0.1.0; touch /tmp/X; :'` reaches the inner shell as inert data
and executes nothing. I also confirmed the mechanism the fix depends on:
`VERSION=probe-1.2.3 nix develop --command bash -c 'echo $VERSION'` prints
`probe-1.2.3`, so `nix develop` does pass the job env through. And if it ever
stopped doing so, `set -u` in the inner shell makes that a loud unbound-variable
failure rather than a smoke test comparing against an empty string.

**MAJOR (published before assets were attached).** The sequence at
release.yaml:244-275 cannot leave a live, empty release. I walked every branch:

- first run, upload fails -> a DRAFT remains. Not visible, nobody notified.
- first run, the final `--draft=false` fails -> a draft WITH assets remains.
- re-run over that draft -> `gh release view` finds it, `gh release edit` does
  not pass `--draft` so it stays a draft, assets clobber, then it flips visible.
  Converges.
- re-run over an already-LIVE release -> notes/title/prerelease are edited on a
  release that is already live and already has assets, then assets clobber,
  then `--draft=false` is a no-op. There is a window where a live release shows
  new notes beside old assets, but no window in which a live release exists
  with nothing on it. That is the property the step claims and it holds.

**MAJOR (docs/ check too strict) and MAJOR (SIGPIPE).** Both fixed, and the
second is verified: I reproduced the round-1 failure and re-ran it against the
new code (check-release-ready.sh:68-77) with 4000 files under `docs/scratch/`.
It now exits 1 with its own diagnostic instead of a bare 141. The comment at
lines 65-67 records why the pipe went away, which is the right place for it.

**MINOR (`--prerelease`).** The coordinator's assumption is correct, and I
checked it rather than taking it: `gh` registers `--prerelease` as a pflag
bool, so `--prerelease=false` parses via `strconv.ParseBool` -
`gh release create ... --prerelease=notabool` fails with
`invalid argument "notabool" for "-p, --prerelease" flag`, at parse time,
before any network call. `gh release edit` only sends a field whose flag was
explicitly Changed, so the `=false` form is what makes clearing a pre-release
marking possible at all; gh's own help for `release edit` uses
`gh release edit v1.0 --draft=false` as its example. Both spellings in the
workflow are right, and a malformed `PRERELEASE` (empty, say) fails loudly
instead of silently defaulting.

**MINOR (PEP 440).** `is_prerelease` is right on everything I threw at it:
`1.0.0.post1`, `1.0.0+deadbeef` and `1.0.0-1` are all false; `0.2.0rc1`,
`1.0.0.dev4`, `v0.1.0-rc.1`, `0.1.0.a1` and `1.2.3.PREVIEW.2` are all true.
Moving this out of the workflow and pinning it with
`test_prerelease_classification_follows_pep_440` is the better fix than the one
I suggested.

The remaining minors and the nits are all handled as described, and
`check-release-ready.sh` still passes for `v0.1.0` and exits 1 for `v9.9.9`.

### Still open

### MINOR the dispatch input validation is newline-tolerant

`.github/workflows/release.yaml:60` validates with
`printf '%s' "$RAW_TAG" | grep -Eq '^v[0-9A-Za-z.+_-]+$'`. `grep`'s `^` and `$`
anchor a LINE, not the string, and grep succeeds if ANY line matches. The
`workflow_dispatch` UI field is single-line, but the dispatch REST API and
`gh workflow run` accept arbitrary strings, so a multi-line input reaches this.
I confirmed it: `RAW_TAG=$'v0.1.0\nversion=9.9.9\nanything at all; rm -rf /'`
passes both the `case` and the `grep`, and then lines 64-67 write all of it
into `$GITHUB_OUTPUT`, which is the standard GITHUB_OUTPUT injection - my probe
landed an extra `version=9.9.9` key alongside the real one.

This is NOT the round-1 injection returning. There is no code execution: every
consumer of these outputs takes them through `env:` or as an action input
(`ref:`), and the guard's version-agreement check still has to pass against the
checked-out `pyproject.toml`. It is a hardening gap on the same input, with a
one-line fix that also removes the `printf | grep` pipe:

    if [[ ! "$RAW_TAG" =~ ^v[0-9A-Za-z.+_-]+$ ]]; then

Bash's `=~` anchors the whole string (no newline flag), and I verified it
rejects exactly the input grep accepted.

### MINOR the three jobs resolve the tag independently

`guard`, `verify` and `publish` each check out `ref: <tag>` separately
(release.yaml:72, 108, 199). Nothing ties the commit `verify` gated to the
commit `publish` builds. A tag force-moved between the jobs - or a re-run days
later against a tag that has since moved - produces a release whose artifacts
were built from a commit the gate never saw, and `verify`'s tagged-commit
assertion cannot catch it because it only proves `verify`'s own checkout is
self-consistent.

Low likelihood; nobody moves a release tag mid-pipeline. But the fix is small
and makes the pipeline's central claim provable: have `guard` emit the resolved
SHA as an output (`git rev-parse "${TAG}^{commit}"` after its checkout), have
`verify` and `publish` check out `ref: <sha>`, and keep the existing assertion
so it now proves "this SHA is still what the tag names" rather than "this
checkout checked out what I asked for".

### NIT the scratch check now guards a drawer nothing fills

`check-release-ready.sh:68-77` and the new AGENTS.md paragraph define
`docs/scratch/` as THE ephemeral drawer. That correctly removes the round-1
false positive, but nothing writes there today: `/lessons` is an external skill
that knows about `LESSONS.md`, not this path, and `docs/` does not exist in the
repo at all. So the epic's DoD-4 scratch clause is now carried by a check that
is vacuous until the convention is wired into the tooling that creates scratch.
That is still strictly better than a check that blocks releases on legitimate
design docs, and AGENTS.md documents the convention so a future `/lessons` run
has somewhere to aim. Worth a line in NOTES.md saying the check is a convention
pin rather than an active guard, so a later reader does not over-trust it.

### NIT confirm gh resolves a DRAFT release by tag on the first real run

The re-run-over-a-failed-draft path depends on `gh release view/edit/upload
"$TAG"` finding a release that is still a draft. The REST endpoint "get a
release by tag name" does not return drafts, so gh must be falling back to a
listing - and its own `release edit` help advertises
`gh release edit v1.0 --draft=false`, which only makes sense if it does. I am
confident but could not prove it without creating a release. If it ever did NOT
resolve, the `else` branch would run `gh release create` again and the API
would happily make a SECOND draft for the same tag, which is the duplication
the DoD forbids. Cheap insurance if you want it: replace the existence probe
with `gh release list --json tagName --jq ...`, which sees drafts
unambiguously. Otherwise just watch this path on the first real re-run and
record the result in NOTES.md, where the two outstanding DoD items already
live.

### What is good in round 2

The fixes are better than the changes I suggested in three places. Moving
pre-release classification into `release_tools.is_prerelease` with a test,
rather than patching the shell regex, puts the rule where the rest of the
version logic already lives and pins it against regression. Naming
`docs/scratch/` in AGENTS.md turns "the guard has an opinion" into "the
repository has a convention the guard enforces". And keeping the clean-tree
check while adding the tagged-commit assertion next to it - with a comment at
release.yaml:113-115 saying plainly which one is vacuous where - is a better
answer than dropping either.

The comment discipline is worth calling out on its own. The TAG DISCIPLINE
block at the top of the workflow, the note on why `--verify-tag` is there, and
the "No `| head` here" comment in the guard script all explain a decision that
a future reader would otherwise be tempted to undo. The TASK.md step now
records what was wrong with the first draft instead of quietly re-ticking, which
is what makes the rest of the record trustworthy.
