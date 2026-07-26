# Retro: T4 - Telegram transport (httpx long-poll)

- TASK: 20260722-222734
- BRANCH: feature/telegram-transport
- REVIEW ROUNDS: 1 (APPROVE, out-of-context; 2 MINOR + 1 NIT addressed in-session)

See TASK.md for what changed and why; this is process only.

## What went well

- Grounded the design in the REAL internal turn path before writing code: read
  `_launch_agent_turn` + `_drain_turn` and confirmed they are `create_app`
  closures, so the bot could not import them and had to be wired from `_lifespan`
  through injected callbacks. That firsthand read (not the spike's paraphrase)
  fixed the architecture up front and made the transport unit-testable with a
  fake orchestrator.
- Confirmed the green base suite BEFORE implementing
  (`check-the-base-suite-before-you-start`), so nothing surfaced as a surprise
  red at verify time.
- The out-of-context reviewer's R1.2 ("the real callbacks have zero coverage")
  turned into a clean win: extracting the closures into a module-level
  `build_telegram_callbacks` made every branch unit-testable AND was the natural
  home for the R1.1 error-handling fix. One refactor closed two findings.
- Applied `format-only-the-files-you-edited-not-whole-dirs` and the diff-scoped
  non-ASCII sweep; the diff stayed focused.

## What went wrong

- The first `telegram_allowed_chat_ids` validator mirrored `project_base_dirs`'s
  `mode="before"` split, but a `list[int]` env value ("123,456") raised
  `SettingsError` at the SOURCE before the validator ran: pydantic-settings
  JSON-decodes a complex (list/dict) env field first. Caught at test time, not up
  front. The fix is `Annotated[list[int], NoDecode]`, which hands the raw string
  to the validator (which then owns both the delimited and JSON-array forms).
- Mirroring `project_base_dirs` was mirroring a LATENT BUG: its colon-separated
  env form has the same source-decode problem and does not actually work via env
  (only via a constructor list). The comment claiming it does is aspirational. I
  trusted the pattern instead of reproducing it through the real channel (env).

## What to improve next time

- When a pydantic-settings field is a list/dict/model and you want to accept a
  non-JSON env string (delimited, custom), a `mode="before"` validator is NOT
  enough - the settings source JSON-decodes complex fields first and errors
  before the validator sees the string. Annotate the field `Annotated[T,
  NoDecode]` and parse ALL forms (delimited AND JSON) inside the validator. New
  ledger lesson added.
- When copying an existing config field's "it also accepts X" pattern, prove X
  works through the INTENDED input channel (env vs constructor) before trusting
  it; a validator that only runs on constructor input silently does nothing for
  env-supplied complex fields.

## Action items

- [x] Added `list-env-field-needs-nodecode-for-a-before-validator` (x1) to
      LESSONS.md (Backend section).
- No follow-up code tasks. T5 (reply rendering polish + examples/ script + full
  app+mock-backend e2e) is the next queued sibling and consumes the `on_message`
  seam this task established. The pre-existing latent `project_base_dirs` colon
  issue is noted but out of scope - file a task only if it bites.
