# Prove Lane 1 with the conversation demo and the chat explainer

- PRIORITY: 96
- TAGS: feature, v0.2.0, lane1, chat, deliverable
- KIND: TASK
- ACTIVITY: WORKING
- GATES: PLAN
- RESOLUTION: -
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-115256, 20260804-115319, 20260804-115320, 20260804-115321

## Story

As the maintainer, I want Lane 1 to end in something I can run and read, so
that "the conversation exists" is a claim I have watched succeed rather than a
row of green test names.

Note what this task is NOT. `examples/chat_conversation.py` already exists from
the first Lane 1 task - the member gate in `tests/test_examples.py` requires it
the moment `packages/chat` appears. This task does not create the example. It
makes the example prove the WHOLE LANE, and writes the explainer.

## Steps

- [ ] Write the two gates first, both red, in `tests/test_examples.py`, beside
      `test_host_report_fixture_calls_every_renderer` and in its shape:
      - `test_chat_conversation_calls_every_exported_function` - parse
        `examples/chat_conversation.py` with `ast`, collect called `Name`s, and
        assert every `types.FunctionType` in `scufris_chat.__all__` is among
        them. Red today on `authorize` alone (probe run 2026-08-04).
      - `test_chat_conversation_renders_an_attributed_causation_tree` - run the
        example as a subprocess (as `test_offline_example_runs` does) and assert
        its stdout carries every `event_seq`, every actor label the demo writes,
        and the tree edges (`main.TREE_GUIDE`-style glyphs) that render
        causation. Red today: the current output is four flat `print` lines.
- [ ] Grow `examples/chat_conversation.py` into the lane demo. `rich` is a ROOT
      dependency (`pyproject.toml:20`) and is importable in the dev shell
      (checked); the example already imports only `scufris_chat` and
      `scufris_core` off `sys.path`, and `rich` is the one third-party addition.
      Render with `rich.console.Console` / `rich.tree.Tree`: `event_seq`, a
      colour per typed actor, causation as a tree under the event it answers.
- [ ] Cover the fourth Lane 1 build task in the demo: mint an `OperatorDecision`
      from the operator's message with `authorize`, and show the refusal
      `authorize` raises for the agent's report. That is what makes the demo
      prove the WHOLE lane rather than three quarters of it (see Notes for the
      overlap with Lane 2's `operator_decision.py`).
- [ ] Keep the backend switch mid-script and re-print with the SAME renderer:
      the semantic transcript is identical, the provider session id is not.
      Both already asserted in `main`; keep both assertions.
- [ ] Put at least one assertion behind every claim the new output makes. The
      example gate judges by EXIT CODE, so a rich table nobody asserts on is
      decoration that still exits 0:
      - render into a `Console(record=True)` and assert on `export_text()`
        before printing it, so the asserted text and the read text are one
        string rather than two renderings that can drift;
      - assert the actor-to-colour map covers `set(ActorKind)` exactly - that is
        the checkable half of "a colour per typed actor", since a piped run
        emits no ANSI at all;
      - assert the tree places the report under the message it answers, not
        merely that both appear.
- [ ] Write `tasks/20260801-154211/chat.html` beside `architecture.html`, reusing
      that file's `:root` tokens and section shape so the two read as one set.
      Content, sourced from the accepted records rather than re-derived: the
      event model; the four owned records (`conversation`, `event`, `delivery`,
      `provider_session`) and who writes each; the settled per-turn granularity
      (`tasks/20260804-115256/DECISION.md` section 1 - one event per meaningful
      thing said, because a turn-grained row cannot answer "who said this" for
      anything inside the turn); and the retention non-decision (same record,
      section on retention - no window, no compaction, the table grows without
      bound, and that is a choice with its reason).
- [ ] Link `chat.html` from `packages/chat/src/scufris_chat/README.md`'s pointer
      list, which already links every record this page compiles.
- [ ] Run the boundary and example gates plus `tatr check`; both gates were green
      on base before the change, so a red one is this task's doing.

## Definition of Done

- The demo exercises the whole lane: every function `scufris_chat` exports is
  called by it, `authorize` included
  (test: `test_chat_conversation_calls_every_exported_function`).
- The operator reads an attributed, ordered transcript with causation as a tree,
  and the demo asserts what it rendered rather than printing decoration
  (test: `test_chat_conversation_renders_an_attributed_causation_tree`).
- `chat.html` states the event model, the four records and their owners, the
  granularity decision and the retention non-decision
  (manual: user reads chat.html and agrees it explains the lane).
- The demo is legible to someone who has not read the code
  (manual: user runs the demo and follows what happened from its output alone).

## Notes

- Lane 1 deliverable of `tasks/20260801-154211/TASK.md`. The lane is not done
  until this record is.
- Depends on all four Lane 1 build tasks.
- Deliberately a separate record: folded into the last build task, this is the
  part that gets dropped under schedule pressure and nobody sees it happen.
- Base state, checked 2026-08-04: `python -m pytest tests/test_examples.py
  tests/test_package_boundaries.py` is 15 passed. Both suites are REGRESSION
  guards for this task, not its proofs - a proof already green on base proves
  nothing - so they are Steps, and the two new tests are the DoD.
- Red-proof probe, run on base in the dev shell: parsing the example's calls
  against the eleven exported functions reports `authorize` missing, and nothing
  else. That is the whole gap between the current example and the lane.
- `authorize` in the Lane 1 demo overlaps Lane 2's planned
  `operator_decision.py`, and that is intended. Lane 2's example is the two
  channel APPROVAL FLOW - asked on two channels, answered on one, the other's
  card resolves. Lane 1's is the one line that closes its own lane: this package
  mints a decision from an operator event and refuses an agent one. The epic's
  Lane 1 sketch predates `20260804-115321` being cut into Lane 1.
- No DECISION.md: nothing here changes a shipped interface. The two choices worth
  naming are in these Notes - the demo covers `authorize`, and the rendering is
  asserted through a recording console rather than through stdout scraping
  inside the example.
- The colour claim cannot be asserted from stdout: `rich` disables colour on a
  pipe, so `test_..._renders_an_attributed_causation_tree` sees plain text. The
  colour is proved by the map covering `ActorKind`, inside the example.
- `examples/` carries no line cap (`scripts/check_file_size.py` covers
  `packages`, `scufris`, `tests`, `web/src` only), so growing the demo needs no
  allowlist entry. `tests/test_examples.py` is well under `TEST_CAP`.
