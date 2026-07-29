# Review: Document the release procedure and cut v0.1.0

- DATE: 20260729-144542
- ROUND: 2
- REVIEWER: out-of-context agent
- VERDICT: APPROVE

(Round 1 was REQUEST_CHANGES; see `## Round 2` at the end for the re-review.)

Scope reviewed: the docs claimed DONE (AGENTS.md `## Releasing`, README `##
Releases` + build block, the NOTES.md runner-side proof, the TASK.md step
ticks). The tagging/publishing steps are correctly still open and were not
judged. All read-only commands in the recipe were executed in this worktree
under `nix develop`; nothing was tagged, pushed or released.

## Findings

### BLOCKER The recipe tags before pushing master, and never says which branch you are on

`AGENTS.md:245-252`. The block is:

```
git commit -am "chore: release X.Y.Z"
scripts/check-release-ready.sh vX.Y.Z
git tag vX.Y.Z
git push origin master
git push origin vX.Y.Z
```

Two independent hazards, both of which end with a published release built from
a commit that is not on master:

1. **No "be on master, be up to date" step.** This repo's own workflow (`/work`,
   `/flow`, sprout) puts a cold session in a WORKTREE on a feature branch - the
   very worktree this review runs in is one. Pasted there, `git commit -am`
   commits to the feature branch, `git tag vX.Y.Z` tags the feature branch head,
   `git push origin master` pushes the (stale, unrelated) local `master` ref,
   and `git push origin vX.Y.Z` then publishes a tag pointing at an unmerged
   commit. Nothing in the guard or the workflow objects: the guard only checks
   version agreement and a clean tree, and the workflow's own assertion is only
   "HEAD is the tagged commit", never "the tagged commit is on master".
2. **Tag created before master is pushed.** If `git push origin master` is
   rejected (non-fast-forward, someone else pushed, no permission), the block
   has no stop condition and the next pasted line pushes the tag anyway. The
   pipeline fires, `gh release create --verify-tag` is satisfied (the tag
   exists), and a release is published for a commit that origin/master does not
   contain. Reversing that means deleting a tag consumers may already have.

Suggested change - make the branch explicit, push atomically, and assert
reachability:

```sh
git switch master && git pull --ff-only     # release from master, up to date
...
git commit -am "chore: release X.Y.Z"
scripts/check-release-ready.sh vX.Y.Z
git tag vX.Y.Z
git push --atomic origin master vX.Y.Z      # both land, or neither does
git merge-base --is-ancestor vX.Y.Z origin/master   # the tag is on master
```

If `--atomic` is not wanted, at minimum reorder to push master FIRST, verify,
and only then tag and push the tag - and say in prose "if the master push is
rejected, STOP; do not push the tag".

### MAJOR The yank block is a paste-runnable fence of three mutually exclusive commands

`AGENTS.md:284-288`. The three lines are alternatives, but only the second
carries an "or" and all three sit in one ```sh fence directly under an
imperative heading. A cold session pasting the block demotes the release,
deletes the release, and then deletes the tag - i.e. performs exactly the
"silently delete it" the paragraph above forbids, and breaks every flake
consumer's ability to re-lock. The third line's guard ("only if nobody can have
fetched it") is unfalsifiable for a public repo: once a tag is pushed, you
cannot know.

Suggested change: split into prose with one recommended path and clearly-marked
escape hatches, or make the alternatives non-runnable (comment them out with a
leading `#`). Recommended path: keep the tag, keep the release, mark it
`[YANKED]` in `CHANGELOG.md`, ship a patch. Deleting a tag should be described
as available only for a tag whose push failed to produce any release.

### MAJOR `gh release edit --prerelease=true` is silently undone by any re-run of the pipeline

`AGENTS.md:285` recommends demoting a bad release with `--prerelease=true`.
That does remove it from "latest", but `.github/workflows/release.yaml` publish
job runs `gh release edit "$TAG" ... --prerelease="$PRERELEASE"` on the
existing release, where `$PRERELEASE` is derived from the version string by
`scripts/release_tools.py prerelease`. For a plain `vX.Y.Z` that is `false`. So
anyone re-dispatching the workflow for that tag (which the section immediately
above tells them to do, and which is advertised as safe idempotence) silently
un-yanks the release. The yank is not durable.

Suggested change: say so explicitly - "the prerelease flag is derived from the
version by the pipeline, so re-running the workflow for this tag will clear it;
the durable record of a yank is the `[YANKED]` marker in `CHANGELOG.md` and the
superseding release." Consider also editing the release notes/title to say
YANKED, which the pipeline overwrites too but at least is visible until then.

### MAJOR The `gh run watch` one-liner watches the wrong run

`AGENTS.md:255`:

```
gh run watch "$(gh run list --workflow release.yaml --limit 1 --json databaseId --jq '.[0].databaseId')"
```

`--limit 1` takes the newest run of that workflow at the instant it is called,
not the run this push triggered. GitHub takes seconds to queue a run after a
tag push, so the normal case is that this resolves to the PREVIOUS run. Run in
this worktree right now it returns `30448350452` - the completed, FAILED
`v9.9.9` negative-proof dispatch recorded in
`tasks/20260729-125101/NOTES.md`. `gh run watch` on an already-completed run
returns immediately and reports that failure, which a cold session will read as
"my release failed" (or, in the mirror case, an old green run read as "my
release passed").

Suggested change: filter by the event/branch and poll for the run, e.g.

```sh
# wait for the run this tag triggered, then watch it
run=""
until [ -n "$run" ]; do
  run="$(gh run list --workflow release.yaml --event push --branch vX.Y.Z \
         --limit 1 --json databaseId --jq '.[0].databaseId')"
  sleep 3
done
gh run watch "$run" --exit-status
```

(`--branch` matches the tag ref for a tag-triggered run.) Add `--exit-status`
so the command's exit code means something. Simpler and safer alternative: tell
the operator to run `gh run list --workflow release.yaml` and watch the run
whose ref is the tag.

### MAJOR The recipe never says to run it inside `nix develop`, and the guard fails outside it

`AGENTS.md:236-256`. Verified: outside the dev shell,
`scripts/check-release-ready.sh v0.1.0` gets two `ok:` lines and then

```
FAIL: tatr not on PATH - run this inside 'nix develop' (it is a flake input)
```

The script's message is good, but the failure lands at step 4 - AFTER the
operator has already made the `chore: release X.Y.Z` commit - so recovering
means entering the shell and re-running, with a commit already sitting there
(and if the guard then finds a real problem, an `--amend` nobody told them
about). `scripts/cut-changelog.sh` and `scripts/release-notes.sh` happen to work
outside the shell, which makes the trap worse: the first three steps succeed.

Suggested change: make `nix develop` step 0 of the fenced block, and add one
line: "if the guard fails, fix the cause and `git commit --amend` rather than
stacking a second commit."

### MINOR "delete the tag ... and tag again" omits the local tag deletion

`AGENTS.md:277-279` says: delete the tag (`git push --delete origin vX.Y.Z`),
fix, and tag again. The local tag still exists, so the next `git tag vX.Y.Z`
fails with `fatal: tag 'vX.Y.Z' already exists` - guaranteed to bite, on the
one path the operator reaches while already dealing with a failure.

Suggested change: `git tag -d vX.Y.Z && git push --delete origin vX.Y.Z`.

### MINOR "Nothing partial is ever visible" is overstated

`AGENTS.md:273-275`. Two ways it is not true:

- The TAG is public the moment it is pushed, which is before the pipeline runs
  at all. A flake consumer pins tags (this same document says so at line 296),
  so a tag whose release never publishes is visible and pinnable. What is
  invisible is the RELEASE PAGE, not the release.
- The draft story holds for the first run only. On a re-run for a tag whose
  release is already published, the workflow takes the `gh release edit` branch
  and mutates a LIVE release in place (notes, prerelease flag, clobbered
  assets). No draft is involved.

Suggested change: "a failure before the last step leaves an unpublished draft
release - the tag is public from the moment you push it, but the release page is
not." And add a sentence that re-running for an already-published tag edits it
live.

### MINOR "re-run the workflow for the same tag" gives no command

`AGENTS.md:275-276`. A cold session must guess between `gh run rerun <id>
--failed` and `gh workflow run release.yaml -f version=vX.Y.Z`, and the second
has a constraint the doc never states: the workflow_dispatch path requires the
tag to ALREADY EXIST (the workflow header says it, the doc does not), and
dispatch only works from the default branch.

Suggested change: give both commands, and note the tag-must-exist constraint.

### MINOR The flake-lock consequence of a moved tag is stated imprecisely

`AGENTS.md:291-293`: "a moved tag changes what their lock file resolved to". A
locked flake input records `rev` + `narHash`, so an ALREADY-LOCKED consumer is
unaffected by a moved tag - it keeps fetching the pinned rev. What actually
happens is: any `nix flake update` / fresh lock silently resolves the same tag
to different code, and a DELETED tag makes locking fail outright while existing
locks keep working. That distinction is the whole reason "prefer a new patch
release" is right, so it is worth getting exact.

Suggested change: "existing lock files keep resolving the rev they pinned; a
moved tag silently changes what the NEXT `nix flake update` resolves, and a
deleted tag breaks re-locking entirely. Note also that deleting a tag does not
un-fetch it: anyone who already locked it still has the commit."

### MINOR README pins a tag that does not exist yet

`README.md:67` and `AGENTS.md:296`. `git ls-remote --tags origin` is currently
empty - there are no tags in this repository at all. If this docs branch lands
before the tag is pushed (and the task's own steps put the tag AFTER the docs),
the README publicly documents a flake input that fails to resolve.

Suggested change: land the doc and the tag in the same push, or verify the pin
resolves as part of the tagging step. Not a change to the text - a sequencing
note in the task is enough.

### NIT The recipe assumes `[Unreleased]` already has content

`AGENTS.md:238-243`. Step 1 is "bump the version", step 2 is "cut". Nothing
tells the operator to WRITE the changelog entries first. `cut_changelog` refuses
an empty `[Unreleased]` with `"[Unreleased] is empty; there is nothing to
release as X.Y.Z"`, which is a fine error but arrives with no instruction. One
sentence ("the entries should already be there - AGENTS.md's Docs sync rule
puts them in with the change") closes it.

### NIT Paste-order in the README build block

`README.md:45-46`: two consecutive `cd web && ...` lines with no `cd ..`. The
second fails if the block is pasted line by line after the first. Same shape
already exists at `AGENTS.md:70-71`, so this is a pre-existing convention rather
than a regression - but the release doc is the one place in this diff that
insists on paste-accuracy, so it is worth making both `(cd web && npm ci)`.

### NIT Pre-release classification is undocumented

The pipeline classifies `v1.0.0rc1` as a pre-release by PEP 440 (and `1.0.0.post1`
as not), which changes the release page and interacts with the yank advice. The
`Releasing` section never mentions that suffixed tags are handled at all.
One line under "What the pipeline does" would do it.

## What is good

- The scripts do what the doc says they do. Verified in the worktree under
  `nix develop`: `scripts/cut-changelog.sh --check 0.1.0` -> `CHANGELOG.md is
  cut for 0.1.0`; `scripts/release-notes.sh 0.1.0` prints the section body;
  `scripts/check-release-ready.sh v0.1.0` prints five `ok:` lines and
  `release-ready: 0.1.0 (tag v0.1.0)`; `rg -n "Releasing" AGENTS.md` -> 221.
- Every claim in "What the guard checks" is true of `check-release-ready.sh`,
  including the "dated, non-empty" nuance (`check_agreement` requires
  `top.date`, `notes` refuses an empty body).
- "What the pipeline does" matches `release.yaml` job for job, including the
  draft-then-flip ordering and the smoke test asserting `scufris $VERSION`.
- The idempotence claim on `cut-changelog.sh` is exactly right, including the
  subtle part (a dateless re-run never moves the date; only an explicit
  `--date` re-dates).
- README's version claims all check out: `--version` exists
  (`scufris/cli.py:44`), the health field is `scufris_version`
  (`scufris/health.py:41`, route `/api/agent/health` at `scufris/app.py:1919`),
  and the settings view renders it (`web/src/settings-view.ts:308`). The
  "task records" addition to the `nix flake check` comment is true
  (`flake.nix:217`), and the CI badge the text refers to is already there.
- Repo conventions hold: the whole diff is plain ASCII (no non-ASCII byte in
  `git diff master...HEAD`), and the commit carries no AI attribution.
- The NOTES.md addendum is honest about what the `v9.9.9` dispatch did NOT
  prove, which is the right instinct and the reason the negative proof is
  trustworthy.

## Round 2

- DATE: 20260729-145600
- COMMIT: c555943
- VERDICT: APPROVE

All thirteen round-1 findings are addressed, and the two structural ones (the
tag/push ordering and the paste-runnable yank fence) are fixed at the root
rather than papered over. Re-verified against `release.yaml` and by running the
commands.

### Round-1 findings: confirmed fixed

- **BLOCKER (ordering / wrong checkout)** - the section now opens with "from the
  MAIN checkout on master, inside `nix develop` - not from a sprout worktree",
  states WHY (a tag on a feature branch publishes a commit master does not
  contain), and opens the block with `git branch --show-current` and
  `git pull --ff-only`. `git push origin master` is step 5 and the tag is step
  6, with the rejected-push rationale in the comment. Following it literally is
  now safe: after a successful master push, `git tag vX.Y.Z` tags exactly the
  commit that landed.
- **MAJOR (yank fence)** - replaced by four labelled prose alternatives, headed
  "These are alternatives, not a sequence", fix-forward first, each carrying its
  own consumer consequence. Nothing is paste-runnable as a chain any more.
- **MAJOR (prerelease not durable)** - stated explicitly, naming the publish
  job's re-classification as what undoes it, and scoped to "a stopgap". Matches
  `release.yaml`'s `gh release edit ... --prerelease="$PRERELEASE"`.
- **MAJOR (wrong run watched)** - both the watch and the rerun one-liners filter
  `--branch vX.Y.Z`, and step 7 lists the runs before watching. Verified against
  this repo: `gh run list --workflow release.yaml --branch v9.9.9` returns
  nothing, while `--branch master` returns the old dispatch run 30448350452 -
  i.e. the filter does exclude the exact run that misled the round-1 version.
- **MAJOR (no `nix develop`)** - in the opening sentence with the reason, plus
  the clean-tree explanation for why the guard runs after the commit and the
  `git commit --amend` recovery.
- **MINOR 6-9, NIT 11-13** - `git tag -d` added; the "nothing partial" claim
  corrected to distinguish the release page from the public tag; `gh run rerun
  ... --failed` added; the lock-file wording rewritten; the empty-`[Unreleased]`
  refusal noted; `cd web` split; pre-release classification documented.
- Pre-release claims verified by running `release_tools prerelease`:
  `0.2.0rc1 -> true`, `1.0.0.dev4 -> true`, `1.0.0.post1 -> false`,
  `0.1.0 -> false`. The doc's wording, including the `post1` exception, is
  exactly right.
- Conventions still hold on c555943: the whole `master...HEAD` diff is plain
  ASCII, no AI attribution in either commit message.

### On round-1 finding 10 (README pins `v0.1.0`)

Accepting the argument. A concrete first version is more useful to a reader
than a `vX.Y.Z` placeholder, and the exposure window is the minutes between this
branch landing and the tag push. Two conditions, neither of which needs a doc
change: this task must not close with the tag unpushed, and the pin should be
proved once the release exists (`nix flake metadata github:alexjercan/scufris/v0.1.0`
resolving is the cheapest proof, and belongs in the task's verification notes).

### Still open (none blocking)

#### MINOR "a fresh clone will fail to resolve it" is not true

`AGENTS.md:336-339`. The parenthetical is right that an existing `flake.lock`
keeps working because it records the commit hash - but a FRESH CLONE of a
consumer repo whose `flake.lock` is committed also keeps working, for the same
reason: nix fetches the locked rev, not the tag. What actually breaks is
re-locking (`nix flake update`, `--recreate-lock-file`) and any consumer adding
the input for the first time.

Suggested change: "...but `nix flake update`, or a new consumer adding the input
for the first time, will fail to resolve it."

#### MINOR `gh run watch` without `--exit-status` exits 0 on a failed run

`AGENTS.md:271-273`. `gh run watch` reports the conclusion on screen but exits 0
regardless, so the command chain gives no machine-readable signal - and a cold
session scripting around it would read success. The following
`gh release view vX.Y.Z` does catch the case in practice, so this is a
robustness nit rather than a correctness one.

Suggested change: append `--exit-status` to the `gh run watch` invocation.

#### NIT The watch one-liner still has a (harmless) race

Verified: if the run has not registered yet the command substitution is empty
and `gh` prints `failed to get run: HTTP 404: Not Found` rather than watching
something wrong or prompting. That is a good failure mode, and the preceding
bare `gh run list` line is the intended manual check. Optional: one clause -
"if this 404s, the run has not been queued yet; re-run it."

#### NIT `--branch vX.Y.Z` does not match a manual dispatch

Verified on run 30448350452: a `workflow_dispatch` run's head branch is `master`,
not the tag. So if a later operator re-releases by dispatching the workflow
instead of using `gh run rerun`, the watch and rerun one-liners will find
nothing. Worth half a sentence, since the workflow does expose a dispatch input.

#### NIT `gh release delete` in the untag block

`AGENTS.md:317`. The comment already says "if one was created", which is the
important part; note only that the command errors (harmlessly) when no release
exists, so it should not be read as part of an unconditional sequence.
