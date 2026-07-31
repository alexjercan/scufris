# Retro: Split the oversized frontend views under the size cap

- TASK: 20260731-171431
- BRANCH: refactor/split-frontend-views
- REVIEW ROUNDS: 1 (APPROVE, 1 MINOR + 2 NITs, all fixed on the branch)

## What went well

**Writing the two verification scripts BEFORE the first split paid for itself
immediately.** The move proof came back 637 = 637 with zero differences on
commit 1, which is a stronger statement than any amount of re-reading, and it
was only possible because the script existed before there was a result to shape
it around. The surface diff caught nothing across four splits - 0 DROPPED every
time - which is exactly what it was written to prove after 20260731-171430 lost
8 public names to the opposite habit.

**The no-facade decision held, and was provable.** The plan bet that flat
siblings plus repointed importers were safe because `tsc` turns a missed repoint
into a build failure. Review tested that by sabotage rather than trusting it:
reverting one import in `agent-view.ts` fails the ts-loader build with one
error. That is the whole safety argument for the shape, and it is now measured.

**Cutting the FUNCTION, not just the file, was the right call for the chat.**
`createAgentChat` was 710 lines on its own; no file-level move could have got
`agent-chat-view.ts` under the cap. Extracting two state-owning collaborators
(`createTurnRunner` owning `streaming`/`cancelCurrent`, the composer owning the
palette and image state) left the component at 511 lines and each collaborator
nameable in one sentence.

**The scope boundary held under pressure.** Both allowlisted `.test.ts` files
sat right there, obviously splittable, and moving their assertions would have
dropped them under 900 - which the guard fails as a STALE entry. Import-line
edits only; `it(` counts identical to `master`.

## What went wrong

**Two prettier flags on moved code, both self-inflicted the same way.** Import
blocks written by hand or by generator get collapsed by prettier, and both times
it surfaced as a gate failure rather than before the commit. The baseline
recorded on the sprout is what made them provably introduced rather than
inherited, so the diagnosis was cheap - but the flag should not have reached the
gate at all. This is `format-before-the-check-gate` for the third time.

**An ALLOWLIST deletion ate the wrong characters.** The `.replace()` needle
carried 4 spaces of indentation against an 8-space-indented line, so it matched
mid-line and left the closing brace misindented. `ruff format` would have caught
it, but what actually caught it was reading `git diff` on the edited file - the
"re-read edited artifacts" rule, earning its place again.

**The close-out over-claimed.** "No logic line moved in any commit" is true of
three commits and false of the chat cut, whose move-proof numbers the record did
not report at all. Review caught it (R1.1). The claim was not wrong by accident:
it generalized from the three commits that WERE pure moves, which is precisely
the shape of over-claim a per-commit table would have prevented.

**A pinned tool was behind its own data format.** `tatr` 0.1.0 reported a false
`unplanned-in-progress` on this record because it read `FLOW STEP`/`PLAN STATUS`
only from a `## Flow State` section while every record in the repo carries them
as header bullets. The temptation was to reshape this one record to satisfy the
linter - which would have split the repo's record format for a tool bug.
Diagnosing it took an experiment (adding the section made it pass; reordering
the bullets did not); fixing it took a pin bump.

## What to improve next time

- Run scoped `prettier --write` on every file a generator touched BEFORE the
  first commit, not after the gate complains. It is one command and it removes
  the whole class.
- Anchor a scripted `str.replace` on the entire line including its indentation,
  and assert the count before replacing. A four-space needle on an eight-space
  line is a silent partial match.
- Report the move proof per commit in a table. A prose summary invites the
  generalization that produced R1.1.
- When a repo-pinned lint disagrees with a record that looks correct, check the
  PIN before touching the record.

## Action items

- [x] R1.1, R1.2, R1.3 fixed on the branch before landing.
- [x] `tatr` pinned to v0.2.0, matching nix.dotfiles.
- [x] Every pending ledger promotion dispositioned; the seven PROMOTEs carry
      against 20260731-233221.
- [ ] 20260731-171432 owns splitting `agent-chat-view.test.ts` (1183) and
      `host-view.test.ts` (997), the two ALLOWLIST entries this task left.
- [ ] 20260731-233221 owns turning the promoted lessons into real guards,
      including the two this task paid for again.
