# Retro: Split the host, hostd, and auth modules under the size cap

- TASK: 20260731-171430
- BRANCH: refactor/split-host-hostd-auth
- REVIEW ROUNDS: 1

## What went well

Measuring the Fog question rather than arguing it. The epic left "is `auth.py`
better trimmed than split" open, and it is the kind of question that invites an
opinion. The answer was arithmetic: 606 against a 600 cap, five lore sites in
the file, each a citation appended to a sentence that keeps its invariant, worth
at most four lines. 602 is still over. The only remaining source was the module
docstring the comment policy says to keep. One measurement closed a question the
epic had carried for three tasks, and it went into DECISION.md as a number
rather than a preference.

The second half of that answer mattered more than the first: even if the sweep
HAD reached 600, clearing a ratchet by one line is not clearing it. That is the
part a purely arithmetic reading would have got wrong.

Four independent modules landed as four commits, and the guard was proved green
at every one with `git rebase master --exec 'python scripts/check_file_size.py'`
rather than asserted. That is the right shape here and the opposite of
20260731-171429's single indivisible commit - the difference being whether the
unit of change is one module or one package that cannot exist halfway.

The verification was written rather than read. Two scripts did the work a second
reviewer would otherwise have done by eye: an `ast`-vs-`dir()` comparison that
lists every public name a facade dropped, and a normalized code-line multiset
difference that shows exactly which lines left and arrived. The second one
proved the claim the whole task rests on - zero logic lines added, removed or
reordered across all four modules, so `OPERATOR_ONLY_PATTERN`, the scrypt
parameters and the 0600 session flush are provably verbatim. For a security
boundary that is worth more than a careful reading, because it does not depend
on the reader noticing.

Two lessons carried in from siblings paid off immediately. The pre-move
`ruff`/`mypy`/`pytest` baseline was recorded at PLAN time, before any code moved,
and all four files were clean - so every later flag was settled in advance as
introduced rather than inherited. And the plan named the `_inspector` patch
targets before the first file was written, so all four were repointed in the
same commit as the split.

## What went wrong

**The facade narrowed the public surface in a way no record declared.** Three
modules lost their `logger` and `scufris.hostconfig` lost five more names
(`MAX_CHANGES`, `MAX_LOG_TAIL`, `GIT_TIMEOUT`, `EVAL_TIMEOUT`, `Propose`). Every
one is correct - nothing outside the package consumes any of them, and the
epic's rule is to export only names with a real consumer - but only the `logger`
drops were written down, and only because they were noticed while writing the
submodule rather than by any check. The other five surfaced in review, from the
`ast`-vs-`dir()` script. The facades were built by asking "what do the callers
import", which finds everything that BREAKS but nothing that silently shrinks.

**A `git reset --hard` discarded uncommitted `tatr flow` state.** Halfway
through, cleaning up after a stray loop, a hard reset threw away the TASK.md
edit that `tatr flow --to WORKING` had made. It surfaced later as `tatr flow`
reporting `PLANNING -> PLANNED` on a task whose work was already committed - the
state machine had rewound because its state lives in a working-tree file.
Recovered by walking it forward and amending, but it is a silent loss: nothing
warns that a reset just rolled back the lifecycle.

**A base-vs-branch comparison nearly compared the branch to itself.** Checking
whether `ruff format` was already failing on `tests/test_host_mcp_server.py`, the
first attempt ran `git checkout master` inside the sprout worktree. That fails -
master is checked out in the main repo - and the surrounding `bash -c` swallowed
it, so both halves of the "base vs branch" comparison ran on the branch and
agreed. The output looked like a clean answer. The real measurement needed
`git show master:<path>` into a scratch file, which does not depend on switching
branches at all.

## What to improve next time

Derive a facade from the BASE module's public names, not from the callers'
imports. `ast`-parse `git show master:<module>` for module-level public names,
diff against `dir()` on the new package, and decide each difference explicitly:
export it, or record why it went. Callers tell you what would break; only the
base module tells you what quietly stopped existing.

Prove a move-only refactor is move-only. Normalize both sides to a multiset of
stripped, non-blank, non-comment, non-import lines and difference them. Every
remaining entry should be a docstring rewording or a rename you can name. It
takes one script and it converts "I moved it carefully" into evidence - which is
what a security boundary in the diff actually needs.

From a sprout worktree, read the base with `git show <base>:<path>`, never
`git checkout <base>`. The checkout cannot succeed while the base is checked out
elsewhere, and inside a compound command the failure is easy to miss while the
comparison it was setting up still prints something that looks like an answer.

Commit the `tatr flow` transition, or re-run it, before any `git reset --hard`
or rebase. The lifecycle state is a tracked file like any other.

## Action items

- None requiring a follow-up task. The three review findings were record and
  docs hygiene, all fixed on this branch before landing.
- The remaining epic children (20260731-171431 frontend views, 20260731-171432
  test suites) inherit the facade-derivation and move-proof checks via
  LESSONS.md.

## Diagnosis

- **Breadth.** Four modules, ~2670 lines, four commits - and correctly four, not
  a missed split: the modules are independent, share no seam, and each commit
  deletes its own allowlist entry so the guard holds at every one. This is the
  20260731-171428 shape rather than the 20260731-171429 one, and the plan chose
  between them deliberately. The diff is large because four things were in
  scope, which the epic decided, not because scope was found late.
- **Churn.** One round, three NITs, no BLOCKER or MAJOR. Only R1.2 (the
  undeclared surface narrowing) had a plan-time question that would have caught
  it, and it is neither of the two the skill names: not the from-scratch
  challenge (the package shape was right) and not the cold-reader rationale test
  (the rationale was sound). It is a missing SURVEY step - "list what each
  facade drops relative to the base module" - which the plan's survey did not
  contain because every prior task's survey was framed around call sites that
  break. R1.1 and R1.3 are work-time hygiene: a README sentence made less
  accurate by an edit, and a lore-site count that fell one behind what landed.
- **Context.** No compaction, no checkpoint crossed, no handoff, no delegation.
  Nothing measured or observed indicated pressure. The four modules were read
  and written one at a time, each package finished and committed before the next
  was opened, so the working set was one module rather than four - which is the
  same property the epic exists to give every future task.
