# Split the Telegram surface under the size cap

- PRIORITY: 85
- TAGS: refactor, v0.2.0, telegram, backend, maintainability
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260731-171411
- DEPENDS ON: 20260731-171420

## Story

As a maintainer, I want the Telegram surface split from one 1448-line module
into transport, command handling, and rendering, so that a bot change does not
load the whole surface.

## Steps

The module becomes a package of the same name (DECISION.md); import paths do not
move, so no caller changes. Submodules import each other directly, never through
their own `__init__`. This lands as ONE commit: a module cannot become a package
incrementally without failing the size guard in the intermediate state.

- [x] Record the baseline BEFORE any move: `python -m pytest` green and its test
      count, `python scripts/check_file_size.py` green, and `ruff format
      --check`/`ruff check`/`mypy` green on `scufris/telegram.py`. A later
      failure on moved code is then this change's, not inherited.
- [x] Create `scufris/telegram/` and move code into it verbatim, in this
      dependency order (`text -> contracts -> render -> api -> turn ->
      approvals -> bot`):
      `contracts.py` (`OnMessageStream`, `OnReset`, `OnCancel`,
      `OrchestratorInfo`, `SettingsOps`, `ApprovalOutcome`, `ApprovalOps`);
      `text.py` (the constants, reply strings, callback codes, emoji, and the
      small formatters `_scrub`, `_fenced`, `_gib`, `_mib_per_sec`, `_fmt_*`,
      `_health_mark`, `_worst_status`, `_hottest_temp`, `_toast`, `_preview`,
      `_command_of`, `_command_arg`); `render.py` (the pure renderers
      `render_reply`, `markdown_reply`, `_format_reasoning`, `_format_tool`,
      `render_stats`/`_health`/`_usage`/`_tools`/`_settings_summary`,
      `settings_markdown`, `render_approval`, `approval_keyboard`,
      `confirm_keyboard`).
- [x] Cut `TelegramBot` at its transport seam into `api.py`: `BotApi` owns the
      httpx client, the base URL, the `getUpdates` offset and the five wire
      calls (`get_updates`, `send_message`, `edit_message`, `send_chat_action`,
      `answer_callback`) plus `_message_id`. Client ownership and close-on-exit
      semantics move with it unchanged.
- [x] Move the streamed-turn message sequence into `turn.py` as functions over
      a `BotApi`: `render_turn`, `send_reply` (both MarkdownV2 fallbacks intact),
      and the typing loop.
- [x] Move the approval surface into `approvals.py`, whole: the bounded
      `_announced` and `_reason_prompts` state, `announce_proposal`,
      `announce_decision`, `send_digest`, `/approvals`, `/deny` and its
      force-reply flow, and `_handle_callback`. Every approval path lands in
      this one module and none of it in `render.py` or `turn.py`. The chat-id
      allowlist stays on `TelegramBot`; it decides whether an update is
      dispatched at all.
- [x] `bot.py` keeps `TelegramBot`: construction and wiring, `run`,
      `poll_once`, `_handle_update`, `_dispatch`, `/settings`, turn-task
      lifecycle, and delegation for the three public approval methods
      `app.py` calls.
- [x] Write `__init__.py` as the facade: `TelegramBot`, the contracts, the
      `render_*` names, the reply-string constants and the four private helpers
      `tests/test_telegram.py` imports (`_command_of`, `_command_arg`,
      `_format_reasoning`, `_format_tool`). Do not add a name to `__all__` to
      silence F401; drop the unused import instead.
- [x] Repoint every test reach into a private attribute that the split moves:
      `tests/test_telegram.py:1102` (`telegram_mod.telegramify_markdown` ->
      `scufris.telegram.render`), and in `tests/test_telegram_approvals.py` the
      `bot._approvals`, `bot._remember`/`_await_reason` and
      `bot._announced`/`_reason_prompts` assertions, which now live on
      `bot._approvals`. There are no string monkeypatch targets naming
      `scufris.telegram`. No `tests/` or `examples/` IMPORT line changes.
- [x] Apply the epic comment policy to every docstring that moves: delete the
      phase-code lore at the four known sites - `pre-T6` (module docstring),
      `Review round 1, R1.1` (`render_approval`), `review round 1, R1.2`
      (`_announced`), `(R1.3)` (`_dispatch`) - keeping each invariant as a fact
      about the code. Introduce no task IDs.
- [x] Delete `scufris/telegram.py` from the guard's `ALLOWLIST` in the same
      commit. Leave `tests/test_telegram.py` alone: 20260731-171432 owns it.
- [x] Update the `scufris/README.md` module-map row (line 334) and the
      architecture diagram label (line 27) for the new layout.
- [x] `git add` the new package BEFORE running `nix flake check`: it evaluates
      only tracked files.

## Definition of Done

- Every file under `scufris/telegram/` is at or under 600 lines and the
  allowlist no longer names the module - 1 hit on base, none after; guard green
  both times
  (cmd: `rg -n "scufris/telegram\.py" scripts/check_file_size.py; python scripts/check_file_size.py`).
- `tests/test_telegram.py` remains over the test cap and remains allowlisted;
  splitting it and deleting that entry belong to 20260731-171432
  (cmd: `rg -n "tests/test_telegram\.py" scripts/check_file_size.py`).
- Command, callback, approval, and rendering behavior unchanged
  (cmd: `python -m pytest tests/test_telegram.py tests/test_telegram_approvals.py`).
- The whole suite passes with the same count as the recorded baseline
  (cmd: `python -m pytest`).
- The import surface did not move: `scufris/app.py` and
  `examples/telegram_bot.py` are untouched, both test files keep their
  `scufris.telegram` import lines, and every changed line outside `scufris/`
  reaches a PRIVATE attribute rather than an import path
  (cmd: `git diff --stat -- tests/ examples/ scufris/app.py`).
- No task ID enters code, Markdown excluded
  (cmd: `rg -n "[0-9]{8}-[0-9]{6}" scufris/ -g '!*.md'`).
- `scufris/README.md` module map matches the new layout
  (cmd: `rg -n "telegram" scufris/README.md`).
- Full backend gate passes, the known tatr-0.1.0 `records` false positive
  excepted while this task is IN_PROGRESS (cmd: `nix flake check`).

## Notes

- Epic: 20260731-171411. Depends on 20260731-171420.
- Package shape, the two class cuts, and the rejected alternatives: DECISION.md.
- Measured groups inside the 708-line `TelegramBot`: wire calls 85, turn
  rendering 131, approvals 225, dispatch and lifecycle 267. One cut is not
  enough (708 - 85 = 623), which is why the class is cut twice.
- Scope boundary settled: this task deletes only the `scufris/telegram.py`
  allowlist entry. `tests/test_telegram.py` (1760) is owned by
  20260731-171432. `tests/test_telegram_approvals.py` is 860 - inside the 900
  test cap and not allowlisted, so nothing to do there.
- Telegram shares orchestrator paths that 20260729-103712 will also touch. Do
  not extract a shared orchestrator service here; that seam belongs to 103712.
- The guard's `ALLOWLIST` ratchets both ways: an entry left behind after its
  file is gone fails the gate.
- `nix flake check` evaluates only git-TRACKED files; an untracked new package
  fails it with a `/build/work/...` path.
- When a check (ruff format, mypy) fires on MOVED code, run it against the
  pre-move file before writing down a cause.
- Do not fold in behavior fixes; file a task instead.
- Assumption: no `CHANGELOG.md` entry. Internal change, no observable behavior,
  setting, or interface change.

## Close-out

**What / why.** `scufris/telegram.py` (1447) became `scufris/telegram/`, eight
modules, largest 472 lines. Import paths did not move: `scufris/app.py` and
`examples/telegram_bot.py` are untouched and no `tests/` import line changed.
Landed as one commit, because a module cannot become a package incrementally -
every intermediate state either leaves a stale `scufris/telegram.py` allowlist
entry or an over-cap `scufris/telegram/bot.py` outside it, and the guard fails
either way. The `ALLOWLIST` entry went in the same commit.

| Module | Lines | Owns |
|-|-|-|
| `__init__.py` | 101 | the facade |
| `contracts.py` | 110 | callbacks, `SettingsOps`, `ApprovalOps`, `ApprovalOutcome` |
| `text.py` | 111 | Bot API limits, callback codes, operator-facing strings, command parsing |
| `turn.py` | 167 | one streamed turn laid out over messages |
| `api.py` | 207 | `BotApi`: the wire calls, the client, the poll cursor |
| `bot.py` | 288 | `TelegramBot`: poll loop, allowlist, dispatch, `/settings` |
| `approvals.py` | 330 | `ApprovalSurface`: every host-decision path |
| `render.py` | 472 | the pure renderers and the keyboards |

**Why the class was cut twice.** 708 of the 1447 lines were `TelegramBot`
itself, so unlike 20260731-171428 the package shape alone did not close the gap.
Measured groups: wire calls 85, turn rendering 131, approvals 225, dispatch and
lifecycle 267. One cut leaves 623 - still over - which is what forced the second.
The two cuts are the ones with their own state: `BotApi` owns the client, base
URL and cursor; `ApprovalSurface` owns `_announced`, `_reason_prompts` and the
`ApprovalOps` it is the only user of. `TelegramBot` keeps the chat-id allowlist,
which is the credential, and decides whether an update reaches a handler at all.

**Two plan corrections, both made while reading the code.**

| Plan said | Landed | Why |
|-|-|-|
| `send_reply` in `turn.py` | `BotApi.send_reply` | `_send_settings` calls it too, and its MarkdownV2-then-plain retry is a delivery policy, not a turn one |
| `_toast` in `text.py` | `approvals.py` | it takes an `ApprovalOutcome` and has exactly one caller, in the approval surface |
| "the typing loop" in `turn.py` | split: `BotApi.try_typing`, `turn._keep_typing` | the swallow-and-log of one failed action is delivery policy; the 4-second re-send loop is turn-lifetime. Found by review, not at write time |

**Difficulties.** The plan named one test line to repoint and there were ten,
across two files. `tests/test_telegram.py:1102` was the known one
(`telegram_mod.telegramify_markdown`, a module-object reach the package
`__init__` no longer binds - it fails LOUDLY with AttributeError, unlike the
silent patch-target class 20260731-171428 hit). The nine others were in
`tests/test_telegram_approvals.py` and the pre-split survey did not report them,
because they are neither imports nor monkeypatch targets: they are direct reads
of PRIVATE attributes that the second cut moved off `TelegramBot` -
`bot._approvals is ...ops` (x2), `bot._remember`, `bot._await_reason`,
`bot._announced` (x3) and `bot._reason_prompts`. All ten now name the object that
owns the state; no assertion was weakened and no import line moved. The DoD
clause was reworded to say "every changed line outside `scufris/` reaches a
private attribute" rather than naming one line.

`ruff format --check` flagged `text.py` after the move. Measured against the
PRE-move file BEFORE writing this down: `scufris/telegram.py` was already not
format-clean at the merge-base, in exactly two places -
`APPROVALS_UNAVAILABLE`'s implicit concatenation and the `acknowledge = (...)`
ternary in `_handle_callback`. Both were carried verbatim, so the flag is
inherited, not introduced. `ruff format` on the package joined both.

**Evidence.**

| Proof | Result |
|-|-|
| `python scripts/check_file_size.py` | green; 1 allowlist hit for `scufris/telegram.py` on base, 0 after |
| `python -m pytest` | 896 passed, same as the recorded baseline |
| `pytest tests/test_telegram.py tests/test_telegram_approvals.py` | 88 passed |
| `git diff --stat -- tests/ examples/ scufris/app.py` | 2 files, 10 lines, all private-attribute reaches |
| `rg -n "[0-9]{8}-[0-9]{6}" scufris/ -g '!*.md'` | no hits |
| `rg -n "R[0-9]\.[0-9]\|pre-T[0-9]\|review round" scufris/telegram/` | no hits |
| `nix build .#checks.{ruff,mypy,pytest,filesize}` | all four green |
| `nix flake check` | `records` reports the known tatr-0.1.0 false positive; the other four checks are the line above |

Every largest-file number is under the 600 cap with room: 472 is the worst.
Four lore sites were swept - `pre-T6` in the module docstring, `Review round 1,
R1.1` in `render_approval`, `review round 1, R1.2` on `_announced` and `(R1.3)`
in the reason-reply path - each keeping its invariant as a fact about the code.
No task IDs entered the package.

**Reflection.** The survey checklist from 20260731-171428 (import sites + string
monkeypatch targets) was necessary but not sufficient: it found zero of the nine
`test_telegram_approvals.py` reaches, because moving a method or a field OFF a
class breaks plain attribute access, which no import-shaped grep reports. The
generalised rule is to grep the private names a split MOVES, not only the module
paths it renames - `rg 'bot\._'` over the test tree would have listed all nine
before the first file was written. These failed loudly, so the cost was one test
run rather than a silent pass; the same reach on a monkeypatch string is the
class that fails silently.
