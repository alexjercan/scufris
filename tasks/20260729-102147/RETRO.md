# Retro: Add the SQLAlchemy transactional engine core

- TASK: 20260729-102147
- BRANCH: fix/sqlalchemy-engine-core
- REVIEW ROUNDS: 5

## What went well

Falsifying every guard before believing it. Both original pool proofs and all
five later guards were sabotaged and confirmed to fail for the intended reason.
That habit is the only reason this task converged: the nesting guard passed its
own tests in four successive versions, three of which had a hole.

Splitting the boundary out as its own task. It landed green with no store on it,
which is exactly what let the guard be rewritten four times without a schema, a
migration and a store cutover moving underneath it. The plan's reason for the
split was proven by what actually happened.

Telling the round-3 and round-4 reviewers to ATTACK the guard rather than verify
it, and naming the specific attacks to try. Rounds 1 and 2 verified and found the
holes they were pointed at; rounds 3 and 4 attacked and found holes nobody had
thought of. Same reviewer, same code, different instruction.

Three review lanes on round 1 rather than one reviewer. The correctness lane
found the deadlock and the partial-corruption gap; the design lane found the
AGENTS.md violation and the speculative exports; the behavior lane independently
re-ran every proof. One reviewer would have found some of that, not all.

## What went wrong

The nesting guard took four attempts. Each version was a correct fix to the
finding in front of it and introduced or left the next hole:

| Version | Holed by |
|-|-|
| process-global `ContextVar[bool]` | refused two databases that cannot contend |
| `ContextVar[Path \| None]` | missed A inside B inside A |
| `ContextVar[frozenset[Path]]` on the given path | missed two spellings of one file |
| entry-snapshot `reset(token)` | poisoned the context on a non-LIFO unwind |

The failed decision, and why it looked sound: a bool was the smallest thing that
expressed "a transaction is open in this context", and at the time exactly one
database existed, so per-database identity looked like speculative generality -
the YAGNI call the repo's own rules ask for. What that missed is that the guard's
whole job is to distinguish one lock from another, so its identity key is not
incidental detail; it IS the feature. YAGNI applies to capabilities, not to the
key an equality check is built on.

Three round-1 findings (R1.1, R1.3, R1.4) were gaps between what the docs
promised and what the code did, on the one task whose stated purpose is that the
follow-ups can trust the boundary without re-deriving it: non-reentrancy was
undocumented, "raises at open" was true only for an unreadable header, and the
sidecar chmod loop never ran on the fresh-open path, so its test was really
proving SQLite's mode inheritance. The plan asked for the API to be recorded. It
did not ask for the recorded API to be tested against the code, and nothing
caught the difference until review.

`tatr flow` was run three times against the wrong checkout. The `cd <worktree>`
at the head of a compound Bash command persists through the rest of it, so a
`tatr flow` appended after a `git commit` acted on the worktree's stale TASK.md
rather than the main checkout's, moving PLANNED -> WORKING twice under a task
really in REVIEWING. Recovered with `git checkout` on the record and
`tatr -r <root>`, but it cost three turns and briefly made the two copies
disagree about the flow state.

## What to improve next time

Before writing a guard that compares identities, enumerate the ways two things
can be the same or different, and write one test per axis first. Here the axes
were: same process vs same instance; the given path vs the resolved file; the
innermost holder vs all holders; and release order. Four axes, four holes, found
one at a time over four rounds when they were enumerable in one sitting.

When a task's deliverable is "a boundary the next tasks can trust", make testing
the RECORDED API against the code an explicit Step. Every one of the three
doc-versus-code findings would have been caught by reading the README section
back against the module asking "is this sentence true?".

Always pass `tatr -r <root>` explicitly. The tool acts on a checkout, this repo
routinely has two, and the shell's cwd inside a compound command is not a
reliable way to choose between them.

## Action items

- 20260801-123345 (filed during review): the two `needs_tatr` project-task tests
  fail identically on master; not this branch's problem, but they make
  `python -m pytest` a gate everyone has learned to read past.
- No follow-up task for the guard itself. It is now covered by six tests across
  four axes and survived a deliberate attack that found nothing.
