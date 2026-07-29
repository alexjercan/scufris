# Retro: document the release procedure and cut v0.1.0

- DATE: 20260729
- TASK: 20260729-125107
- REVIEW ROUNDS: 2 (REQUEST_CHANGES, then APPROVE)

## What went well

- **The release was boring.** Guard green on the first try, pipeline green end
  to end, artifact runs, tag resolves. Everything that went wrong in this epic
  went wrong in review, before anything was published - which is the whole
  point of putting three tasks in front of this one.
- **Verifying from outside the pipeline.** Downloading the wheel from the
  release page and running it, and resolving the flake pin with
  `nix flake metadata`, are checks a green run cannot fake. The re-run through
  `workflow_dispatch` proved idempotence AND exercised the code path that had
  carried the round-1 blocker in the previous task.
- **Writing the procedure as commands, then following them literally.** That is
  what surfaced the blocker below: prose would have hidden it.

## What went wrong

- **The documented procedure had a blocker in it.** It tagged BEFORE pushing
  master and never said which checkout or branch to run from. Followed
  literally from a sprout worktree - which is where this session does all its
  work - it would have tagged a feature-branch commit and published a release
  for a commit master does not contain. I wrote the steps in the order I
  happened to think of them rather than the order that is safe to fail in.
- **A copy-pasteable fence of mutually exclusive commands.** The yank section
  listed demote, delete-release and delete-tag as three consecutive lines in one
  `sh` block. Anyone pasting the block would have deleted the tag and broken
  every flake consumer pinned to it. A code fence is an instruction to paste;
  alternatives must be prose.
- **A one-liner that watched the wrong thing.** `gh run list --limit 1` resolved
  to whatever ran most recently - on the day I wrote it, the FAILED v9.9.9
  dispatch. A cold session following the doc would have concluded its release
  had failed.
- **Stale remote branches left behind.** Two branches pushed for CI evidence
  survived their `sprout land`, because landing squash-merges and only cleans up
  locally. One still carried a temporary probe workflow with an `on: push`
  trigger. The operator found them, not me.

## Lessons

- `write-a-procedure-in-failure-order-not-thought-order`: order the steps of a
  documented procedure so that the LAST irreversible action comes after every
  check and every reversible one - push the branch before tagging it, not
  after. Then read it as a stranger would: which checkout am I in, which branch
  am I on, which shell? Omitting the context is how a correct sequence becomes
  a wrong one.
- `alternatives-are-prose-not-a-code-fence`: a fenced block of shell reads as
  "paste me". Never put mutually exclusive options in one fence - especially
  destructive ones. List them as prose with the consequence of each spelled
  out.
- `filter-a-gh-run-lookup-by-what-you-actually-want`: `gh run list --limit 1`
  means "the most recent run of anything", which in a repo with dispatches and
  several workflows is rarely the run you mean. Filter by `--branch <tag>` or
  `--event`, and use `--exit-status` with `gh run watch` so a red run fails the
  command instead of exiting 0.
- `delete-a-branch-you-pushed-for-evidence`: `sprout land` squash-merges and
  removes the LOCAL branch, so a branch that was pushed to origin (for a PR, for
  a CI probe) survives, never shows as merged, and can carry temporary workflow
  files. Deleting it is a separate, explicit act at the end of the task.

## What to do differently next time

For anything written as instructions - a release procedure, a runbook, a README
recipe - review it by executing it in the target environment, not by reading it.
Three of the four findings here were things that read perfectly and behaved
wrongly: the ordering, the watch one-liner, the paste-runnable yank block. The
fourth (the missing `nix develop`) would also have surfaced in the first minute
of an honest dry run.
