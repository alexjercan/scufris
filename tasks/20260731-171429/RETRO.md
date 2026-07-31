# Retro: Split the Telegram surface under the size cap

- TASK: 20260731-171429
- BRANCH: refactor/split-telegram-surface
- REVIEW ROUNDS: 1

## What went well

Measuring the class before choosing the shape is what made the plan hold. The
sibling task's package-facade shape was handed over as the epic default, and it
was necessary but not sufficient here: 708 of the 1447 lines were `TelegramBot`
itself, so extracting the constants, contracts and renderers still left the
class over the cap. Counting the four method groups at plan time (85 / 131 / 225
/ 267) turned "how do we cut the class" into arithmetic - one cut leaves 623,
which is over, so a second cut is forced - and named which two seams before any
code moved. Both landed where the plan put them.

The two cuts were not size tricks. `BotApi` is the only reason `bot`, `approvals`
and `turn` can be separate modules at all: transport is called from all three, so
without it the three could only be separated by passing the bot around, which is
mixin coupling under another name. `ApprovalSurface` collects state that was
already a set - `_announced`, `_reason_prompts` and the `ApprovalOps` nothing
else used.

Two lessons carried from 20260731-171428 paid off directly. The pre-move
`ruff format --check` was run BEFORE any code moved, so when the formatter fired
on the new package the record could say "inherited" and name the two exact sites,
with the measurement already in hand. And the facade was written to export only
names with a real consumer: `ApprovalSurface` was imported into `__init__.py`,
then removed once nothing outside the package wanted it.

## What went wrong

The pre-split survey found one test line to repoint and there were ten. The
survey checklist inherited from 20260731-171428 was "import sites plus string
monkeypatch targets", and it correctly found the one module-object reach
(`telegram_mod.telegramify_markdown`). It found zero of the nine reaches in
`tests/test_telegram_approvals.py`, because those are neither imports nor patch
strings: they are plain attribute reads - `bot._approvals`, `bot._remember`,
`bot._announced`, `bot._reason_prompts` - that broke because the SECOND cut moved
those members off `TelegramBot` onto `ApprovalSurface`.

The checklist seemed sufficient because the previous refactor's silent-failure
mode was a patch string, and the fix for it was written as "grep the patch
strings too". That framing is about the module PATH being renamed. It has nothing
to say about a member being moved off a class, which is what the second cut did
and what the previous task never did. The cost was one test run - these fail
loudly with AttributeError or a failed identity assertion, unlike the patch-string
class that fails silently - but the plan and the DoD both asserted "one changed
line outside `scufris/`" and had to be corrected after the fact.

Review round 1 found no BLOCKER or MAJOR, and three of its four NITs were
hygiene on code written during the split rather than design problems: two
functions that lost their underscore while moving from methods to module
functions, and a private constant left in the module the plan assigned it to
rather than the module that turned out to be its only reader.

## What to improve next time

Grep the private NAMES a split moves, not only the module paths it renames.
`rg 'bot\._' tests/` would have listed all nine reaches before the first file was
written. The general form: after deciding which members move off which object,
grep each moved private name across the test tree - the survey is about members,
not modules.

When a split turns a method into a module-level function, decide its visibility
from its callers at that moment, not from the name it had as a method. Three
functions dropped their underscore in the move and only one of them had a caller
outside the module.

Assign a private constant to the module that READS it, not to the module its
category suggests. `_ELIDED` went to `text.py` because it is a string; its single
reader is `render.render_approval`, and `text.py`'s own comment had to point at
that function to explain it - which was the tell.

## Action items

- None requiring a follow-up task. The four findings were record and naming
  hygiene, all fixed on this branch before landing.
- The remaining epic children (20260731-171430 host/hostd/auth,
  20260731-171431 frontend views) inherit the widened survey rule via
  LESSONS.md.

## Diagnosis

- **Breadth.** One module, 1447 lines, one commit - and correctly one, not a
  missed split: a module cannot become a package incrementally without failing
  the size guard in the intermediate state, because any halfway point leaves
  either a stale `scufris/telegram.py` allowlist entry or an over-cap
  `scufris/telegram/bot.py` outside it. This is the structural difference from
  20260731-171428, which split four independent modules and landed four commits.
  The diff is large because the unit of change is indivisible, not because scope
  was found late.
- **Churn.** One review round, all NITs. R1.1 (a third undeclared plan
  correction) is the only one a plan-time question could have caught, and only
  indirectly: the plan wrote "the typing loop" as one thing, and it turned out to
  be two - a best-effort send and a 4-second re-send loop - that belong to
  different owners. The from-scratch challenge would not have surfaced that;
  reading the two functions would have. R1.2 through R1.4 are `work`-time naming
  decisions, not plan-time design ones.
- **Context.** No compaction, no handoff, no checkpoint crossed. The whole
  1447-line module was read in two passes and the eight new modules written
  without re-reading it. The single-commit structure meant there was no
  intermediate state to hold; the risk it creates is the opposite one - a
  compaction mid-split would have had the entire package uncommitted - and it did
  not materialise.
