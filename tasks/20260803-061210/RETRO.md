# Retro: Clear the round-2 findings from the router extraction

- TASK: 20260803-061210
- BRANCH: refactor/router-round2
- REVIEW ROUNDS: 1

## What went well

One review round, APPROVE, two NITs and no rework. The plan was five named
findings inherited verbatim from `tasks/20260801-100425/REVIEW.md` round 2,
each already carrying its own file, line and confirmation, so working it was
transcription rather than discovery.

R2.2 is the piece worth repeating. The plan did not just ask for three tests,
it specified how to prove each one: delete the `raise` the case claims to pin,
watch it go red, restore. Both mutations reproduced independently at review
time. A characterization test written without that step would have been
indistinguishable from a test that asserts nothing.

Scoping the R2.3 grep proof to `DECISION.md` at plan time was the other good
call. The inherited proof covered the whole task folder, which could only go
green by rewriting the REVIEW and RETRO records that quote the stale numbers as
their finding - a proof satisfiable only by falsifying a record is a broken
proof, and it was caught before the work started rather than after.

## What went wrong

The plan named `tests/test_domain_routers.py` as R2.2's home without checking
it against the 900-line test cap. Three cases put it at 933, the suite went red
on `test_check_file_size.py`, and the fix was a new
`tests/test_route_iteration.py`. Cheap to recover from - the check named the
file, the count and the cap - and the forced split is the better arrangement
anyway, since the cases are about `iter_routes`'s contract rather than about
the domain routers. Still a plan-time miss: the file was chosen by where the
neighbouring rigs live, not by what the tests are about.

Separately, a review-side mistake worth recording because it nearly landed as a
silent revert. The recording pass re-derived the red-on-base claim by
`cp -r`-ing the worktree to `/tmp` and restoring master's `routes.py` there. A
copied worktree keeps its `.git` file pointing at the original gitdir, so the
copy shares the real branch's index: `git checkout master -- <path>` staged
master's blob against `refactor/router-round2`, and the next `git commit` swept
it in. The review commit reverted R2.1 while claiming to approve it. Caught by
reading `git show --stat` on a commit that should have touched one file, and
fixed by re-adding the working-tree copy and amending. The working tree was
never wrong; only the index was.

## What to improve next time

Choose a test's file by the subject it pins, not by proximity to related rigs,
and check the destination against the size cap while planning - both are
mechanical questions answerable before any code moves.

For a scratch copy of a git worktree, use `git worktree add` or a clone, never
`cp -r`. A `cp -r` copy is not isolated: it shares the index and every ref of
the source. Verification that mutates tracked files needs isolation git
recognizes as isolation.

Verify commits by content, not by the `git add` that preceded them. `git commit`
writes the whole index, so a file staged from anywhere in the session lands in
it; `git show --stat` against the intended file list is the check that catches
that, and it costs one command.

## Action items

- Neither NIT blocks. R1.1's line-count error is corrected in TASK.md; R1.2
  (raw `WebSocketRoute` versus the `@app.websocket` decorator) is declined with
  its reason recorded in REVIEW.md.
- No follow-up task. The `Mount` skip is the only remaining untested branch of
  `iter_routes`, pre-existing and exercised only when the web dist exists; the
  two lane-D siblings (`20260801-100441`, `20260729-103712`) now have the
  corrected shape to copy.

## Landing message

```
refactor(api): fail closed on websocket routes in iter_routes

Drop WebSocketRoute from the iter_routes skip tuple, so a websocket is
refused rather than silently dropped from the boundary sweep. The skip was
the fail-open hole closed for every other node kind, still open for the one
kind BaseHTTPMiddleware cannot cover: a websocket endpoint added later would
have bypassed enforce_auth with no guard reporting it. The app registers no
websocket today, so nothing regresses and the first one added raises.

Add tests/test_route_iteration.py, falsifying all three of the helper's
refusals - the new websocket case plus the two pre-existing guards that had
never been shown red. Correct the stale line and test counts in the sibling
decision record, and narrow the auth rig's fake-session docstring to what it
actually honours.
```
