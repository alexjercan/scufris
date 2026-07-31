# Decision: split the Telegram surface into a package, cutting the bot at its transport and turn seams

- STATUS: ACCEPTED
- DATE: 2026-07-31
- TASK: 20260731-171429
- TAGS: refactor, maintainability, kiss, telegram
- EPIC: 20260731-171411

## Context

`scufris/telegram.py` is 1447 lines. Sibling task 20260731-171428 established
the epic's shape for a module this size: the module becomes a PACKAGE of the
same name with a facade `__init__.py`, import paths do not move, and submodules
import each other directly. That shape applies here unchanged - the import
surface is small (`scufris/app.py`, `examples/telegram_bot.py`, two test files)
but names private helpers (`_command_of`, `_command_arg`, `_format_reasoning`,
`_format_tool`), so moving paths would churn callers for nothing.

What does NOT carry over is the sizing. `sessions`, `agent`, `backends` and
`agent_store` were collections of functions and several classes; splitting them
by ownership got every piece under the cap. Here 708 of the 1447 lines are a
SINGLE class, `TelegramBot`. Extracting the module-level constants, contracts
and renderers leaves that class over the cap on its own, so the load-bearing
question is where to cut the class.

Measured method groups inside `TelegramBot` (708 lines including the class
header and `__init__`):

| Group | Lines |
|-|-|
| Bot API wire calls (`_get_updates`, `_send_message`, `_edit_message`, `_send_chat_action`, `_answer_callback`) | 85 |
| Turn rendering (`_render_turn`, `_send_reply`, `_keep_typing`, `_try_typing`) | 131 |
| Approvals (announce, digest, `/approvals`, `/deny`, `_handle_callback`) | 225 |
| Dispatch and lifecycle (`__init__`, `run`, `poll_once`, `_handle_update`, `_dispatch`, `_handle_settings`, turn-task tracking) | 267 |

## Decision

`scufris/telegram.py` becomes `scufris/telegram/`:

| Module | Contents |
|-|-|
| `__init__.py` | facade: the public surface plus the four private helpers the tests import |
| `contracts.py` | `OnMessageStream`, `OnReset`, `OnCancel`, `OrchestratorInfo`, `SettingsOps`, `ApprovalOutcome`, `ApprovalOps` |
| `text.py` | the constants, reply strings, callback codes, emoji, and the small formatters (`_scrub`, `_fenced`, `_gib`, `_fmt_*`, `_toast`, `_preview`, `_command_of`, `_command_arg`) |
| `render.py` | the pure renderers: `render_reply`, `markdown_reply`, `render_stats`/`_health`/`_usage`/`_tools`/`_settings_summary`, `settings_markdown`, `render_approval`, the two keyboards |
| `api.py` | `BotApi`: the httpx client, base URL, `getUpdates` offset, and the five wire calls |
| `turn.py` | driving one streamed turn's message sequence: `render_turn`, the typing loop, `send_reply` |
| `approvals.py` | the approval surface: announced-message and reason-prompt state, `announce_*`, `send_digest`, `/approvals`, `/deny`, `_handle_callback` |
| `bot.py` | `TelegramBot`: construction, the poll loop, update dispatch, `/settings`, turn-task lifecycle |

The class is cut TWICE, at the transport seam and at the approvals seam.
`BotApi` is a state-owning collaborator (client, base URL, token, offset), not a
mixin; `approvals.py` holds the approval state (`_announced`,
`_reason_prompts`) and the `ApprovalOps` dependency it is the only user of.
`TelegramBot` keeps the chat-id allowlist and delegates.

This lands as ONE commit. A module cannot become a package incrementally: any
intermediate state either leaves a stale `scufris/telegram.py` allowlist entry
or puts an over-cap `scufris/telegram/bot.py` outside the allowlist, and the
guard fails either way. The epic's one-commit-per-boundary rule applied to
20260731-171428 because it split four independent MODULES.

## Rationale

- The task's own Steps name the seams: "transport, command handling, and
  rendering". `api.py` is transport, `bot.py` + `approvals.py` are command and
  callback handling, `render.py` + `turn.py` are rendering.
- "Keep approval and host-operation flows in one place; they are a security
  boundary, not a rendering concern" is satisfied by `approvals.py`: every
  approval path lands in ONE module, and none of it lands in `render.py` or
  `turn.py`. The chat-id allowlist - the actual credential - stays on
  `TelegramBot`, which is what decides whether an update is dispatched at all.
- Transport is called from all three of dispatch, approvals and turn rendering.
  Without `BotApi` those three cannot be separated without passing the bot
  itself around, which is the mixin coupling under a different name.

## Consequences

- Call sites inside the package change shape (`self._send_message(...)` ->
  `self._api.send_message(...)`). That is mechanical and wide; the two Telegram
  test files are the proof it is behavior-preserving.
- `tests/test_telegram.py:1102` patches `telegram_mod.telegramify_markdown`, a
  module-object reach through the package. After the split the package
  `__init__` will not bind that name, so it must be repointed to
  `scufris.telegram.render`. It fails LOUDLY (AttributeError), unlike the silent
  patch-target class 20260731-171428 hit; there are no string monkeypatch
  targets naming `scufris.telegram` anywhere in the repo.
- Task records and LESSONS.md entries citing `telegram.py:<line>` no longer
  resolve. They are history and are not rewritten.

## Alternatives considered

- **One cut only (transport), leaving approvals on the bot.** Rejected on the
  measurement: 708 - 85 = 623, still over the cap. A second cut is forced; the
  approvals seam is the one with its own state.
- **Cut the turn group out of the class instead of approvals.** Viable on size
  (708 - 85 - 131 = 492) but the turn code is already leaving the class as
  `turn.py` free functions; that cut is taken. Approvals is the additional one.
- **Mixin classes (`_ApprovalsMixin`, `_TransportMixin`).** Rejected for the
  same reason 20260731-171428 rejected a base/subclass split: inheritance keeps
  every line of the coupling it claims to break.
- **Fold `text.py` into `render.py` and `turn.py` into `render.py`.** Rejected:
  the constants are read by `bot.py` and `approvals.py`, not just renderers, and
  the merged `render.py` would be ~551 lines mixing pure formatting with async
  send orchestration - under the cap with no headroom and a worse seam.
- **Extract a shared orchestrator service.** Out of scope by the task's Notes;
  that seam belongs to 20260729-103712.
