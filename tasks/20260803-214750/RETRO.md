# Retro: Delete the legacy JSON import, split the singular agent surface, squash to one baseline revision

- TASK: 20260803-214750
- BRANCH: refactor/split-agent-surface-squash-baseline
- REVIEW ROUNDS: 2

## What went well

Planning overturned the task's own premise before any code moved. The task said
"delete `/api/agent/*`"; reading the callers found twelve of sixteen routes are
the operator console's only door, and D1 split the surface instead of deleting
it. A demolition task that had been executed as written would have shipped a
broken settings page.

D6 demanded the `examples/host_agent.py` re-point be proven by RUNNING the
example rather than by grep, and that is the only reason the wrong patch target
was caught: `scufris.api.agent_runs` does import `get_backend`, so a grep
confirms the wrong answer. The real launch site is
`scufris/orchestrator/runs.py:201`.

Review round 1 was run out-of-context in three lanes, and the lane split earned
its cost - the behavior lane's independent return produced R1.10 plus the
process signal that the doc sweep needed a by-concept pass, neither of which the
other two lanes reached.

## What went wrong

**The squash silently disarmed a test and the first fix hid it.** Deleting five
revisions for one baseline removed the only "behind head at a revision this
build knows" state the tree could build, which is what
`test_the_backup_is_taken_on_the_real_migration_path` stood on - the only test
reaching `backup_database` through `upgrade_to_head`. The first attempt put a
`pytest.skip` in front of it. That seemed sound at the time because the skip is
conditional in shape (`if head has no parent`) and reads as defensive; on this
branch the condition is unconditionally true, so the proof was retired, not
guarded, and `migrate.py:241` became deletable with the suite green. Round 1
caught it (R1.1); the `behind_head` fixture and D7 are the real fix.

**The doc sweep was planned by symbol and line number.** The Step at
`TASK.md:123` enumerates ten line numbers in `scufris/README.md`. That form
catches every stale SYMBOL and passes cleanly over three false CONCEPT claims -
the `ConsoleDeps` dependency list, the "same `AgentRunService` and
`AgentDiagnostics` instances" sharing sentence, and the delegation paragraph -
which became R1.2, R1.3 and R1.4. The DoD grep passing is necessary and not
sufficient, and the plan encoded the grep-shaped sweep as the whole job.

**Prose rewrites went in without a reflow pass, twice.** R1.9 was a ragged wrap
from the round-0 edit; the round-1 rewrites that fixed R1.3 and R1.4 left three
more, which is R2.2. R2.4 is the same class: one of the two gate enumerations
was updated for the new `ruff-format` check and still omits `filesize`.

**One stale tick was fixed and its twin was not.** R1.10 established the
`DELIVERED DIFFERENTLY:` remedy for a ticked Step whose text does not describe
what landed, and it was applied to the web-test Step only. The
`examples/host_agent.py` Step carries the same defect and is R2.1.

**The gate could not see the defect it was supposed to catch.** `ruff check`
does not run the formatter, and `line-length` is configured for the formatter,
so an over-long line reached review green (R1.5). Fixed by adding
`ruff-format` to `flake.nix`.

## What to improve next time

**Breadth.** ~2600 deletions over 50 files in one commit, and it is three
separable demolitions: the alias split, the JSON-import deletion, and the
revision squash. The first two genuinely share `scufris/README.md` and the DoD
grep, so bundling them is defensible. The squash does not - it touches
`db/migrations/`, `db/migrate.py` and `test_db_migrations.py` and nothing the
other two touch. It was independently landable and would have carried R1.1 and
D7 on their own, much smaller, review. The plan encoded three demolitions as one
task because they share a TAG, not because they share a boundary.

**Churn.** The plan-time question that would have prevented R1.1 is not in
`plan`'s from-scratch challenge; it is narrower and worth asking of any deletion
Step: *which existing proofs depend on the state this deletion removes?* Five
revisions were an input to a test, not just history. The question that would
have prevented R1.2-R1.4 is the one the round-1 process signal already names:
sweep the prose by CONCEPT before sweeping by symbol, because a line-number
enumeration is a by-symbol sweep wearing a plan's clothes.

**Context.** Observed pressure was in review dispatch, not implementation: in
round 1 the behavior/proofs lane returned AFTER the round was written and
committed, forcing an amendment commit (`c940614`) and leaving that lane's
severities out of the canonical ranking - it had put R1.1-R1.4 at MINOR where
the recording pass put them higher. Round 2 waited for both lanes before
writing, which is the correct order. Lanes are a barrier: aggregate only when
all of them are in.

## Action items

- Add to any deletion Step: name the proofs that stand on the state being
  deleted, and say whether each is re-pinned or deliberately retired. A retired
  proof needs a DECISION entry, not a skip.
- Sweep doc prose by concept before by symbol; a line-number list in a Step is
  a by-symbol sweep and does not discharge the concept pass.
- After any prose edit, reflow the paragraph and re-read every enumeration the
  edit touched for a second member that also changed.
- When one review round establishes a remedy shape, grep the record for every
  other instance of that shape before responding (R1.10 -> R2.1).
- The `ruff-format` gate added under R1.5 is a repository-wide change recorded
  only in REVIEW.md and the close-out. Future gate additions belong in
  DECISION.md.
- Follow-up: the four open findings from round 2 (R2.1 MINOR, R2.2-R2.4 NIT)
  are non-blocking and unfixed. Land them with the next touch of these files or
  file them; they are recorded in REVIEW.md round 2.
- Blocked, not carried by this task: DoD proof `python examples/host_agent.py`
  stays unreadable until `20260804-041340` lands.

## Landing message

```
refactor(api,db): split the singular agent surface, delete the JSON import, squash to one baseline

Three compatibility surfaces removed from the tree the carve's packages will
be built on.

The four `/api/agent/*` routes that were pure aliases of an
`/api/agents/orchestrator/*` twin - usage, memory, account, health - are gone.
The twelve that survive are the operator console's only door, so they keep
their URLs and move out of a module named `legacy_`: `api/legacy_agent.py` ->
`api/console.py`, `LegacyAgentDeps` -> `ConsoleDeps`,
`build_legacy_agent_router` -> `build_console_router`. The web console's one
caller of a deleted alias is re-pointed.

The pre-database JSON import path is deleted outright - `db/legacy/`,
`import_legacy_state`, `LegacyImportRefused`, the `legacy_import` table,
`examples/state_migration.py` and its fixtures. A leftover `projects.json` is
now ignored entirely.

The five shipped Alembic revisions are squashed into one autogenerated
baseline over the twelve surviving tables. A pre-v0.2.0 database is refused
with a message naming the real cause rather than the "written by a newer
version" one it would otherwise hit; existing databases are not carried
forward.

`flake.nix` gains a `ruff-format` check, which is what the lint-only gate was
missing.
```
