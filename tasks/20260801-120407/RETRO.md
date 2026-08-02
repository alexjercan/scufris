# Retro: Import legacy JSON state into the database

- TASK: 20260801-120407
- BRANCH: fix/legacy-json-import
- REVIEW ROUNDS: 2

## What went well

The sabotage-before-declaring-green habit carried over from the previous task
and paid a second time: five sabotages, five intended failures, and the
transaction sabotage changed the implementation rather than the test - it is
what turned "validate the whole file, then insert" into "insert as each record
validates", which is the only version where the rollback proof proves anything.

The plan's ordering call held up. Landing the import before the cutover, with
an explicit scope fence that nothing under `scufris/` calls it, meant the
review could check one thing at a time and the branch could land green without
an operator seeing any change.

The predecessor's reflection asked for the pre-migration backup to be proven on
the real path once a second revision existed. This task is that revision and
the debt was paid in it, unprompted.

## What went wrong

One MAJOR finding, and it was in prose rather than behaviour: the refusal for a
damaged legacy file told the operator to restore `<name>.pre-sqlite.bak`, which
on that path is always a byte-identical copy of the damaged file the same run
had just made. Restoring it is a no-op, or an overwrite of the operator's own
repair.

Two causes, both structural rather than careless:

- The sentence was copied from DECISION.md section 4, under a Step that said to
  implement the policy table "verbatim". That decision was written before the
  backup was known to be taken unconditionally BEFORE the parse, so its remedy
  was already stale when the Step demanded it be reproduced word for word. A
  verbatim instruction transfers the earlier document's errors along with its
  intent.
- No proof asserted the message's CONTENT beyond the path, line and column, so
  the sabotage pass could not reach it. Every sabotage was green-to-red on
  behaviour; a wrong remedy stays green under all of them.

Smaller: the symlinked-backup refusal raised a bare `RuntimeError`, copied from
`migrate.py`, where nothing catches a narrower type. Copying a neighbour's
error type without asking who catches it.

## What to improve next time

- When a Step says to implement an earlier DECISION verbatim, check each clause
  against the code that now implements it before copying it. A remedy is only
  valid if the artifact it names is actually a repair.
- Give operator-facing failure text its own assertion, not just a
  path/line/column check. The message is a contract with a human under stress
  and it is the one part of a module that no behavioural sabotage can falsify.
- Ask who catches an exception before reusing a neighbour module's type.

## Action items

- 20260802-191034 - repair the two `tatr`-shelling project task tests. Found in
  review (R1.4): both fail identically on `master` and are skipped under
  `nix flake check`, which is why the drift went unnoticed and why this task's
  `python -m pytest` DoD needed an exception.
- The cutover task (20260801-120412) is the first caller of
  `LegacyImportRefused`; it should catch that type and surface the message to
  the operator unaltered, since the message is now the whole remedy.

## Process notes

- No context pressure: no compaction warning, no checkpoint, no delegation. The
  review, the fixes and this retro ran in one session.
- The main checkout's TASK.md read `FLOW STEP: PLANNED` while the worktree read
  `REVIEWING`. `sprout ls` before `tatr show` is what caught it, exactly as
  `flow/resume.md` prescribes; no work was redone.
- Round 2 was `in-session` rather than out-of-context: subagent delegation was
  off for the session. Recorded on the round.

## Knowledge

Both observations went to the central repo as occurrences on existing lessons
rather than new slugs:

- `verification/a-green-gate-has-a-bounded-claim` - sabotage bounds its claim to
  behaviour, so operator-facing failure text stays green under all of it.
- `docs/update-restatements-with-the-source` - a partial-supersede banner
  enumerates what its author noticed; a clause it calls unchanged can still be
  stale against the code.

`knowledge check` clean.
