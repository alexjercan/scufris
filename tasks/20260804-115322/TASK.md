# Prove Lane 1 with the conversation demo and the chat explainer

- PRIORITY: 96
- TAGS: feature, v0.2.0, lane1, chat, deliverable
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
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

- [x] Write the two gates first, both red, in `tests/test_examples.py`, beside
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
- [x] Grow `examples/chat_conversation.py` into the lane demo. `rich` is a ROOT
      dependency (`pyproject.toml:20`) and is importable in the dev shell
      (checked); the example already imports only `scufris_chat` and
      `scufris_core` off `sys.path`, and `rich` is the one third-party addition.
      Render with `rich.console.Console` / `rich.tree.Tree`: `event_seq`, a
      colour per typed actor, causation as a tree under the event it answers.
- [x] Cover the fourth Lane 1 build task in the demo: mint an `OperatorDecision`
      from the operator's message with `authorize`, and show the refusal
      `authorize` raises for the agent's report. That is what makes the demo
      prove the WHOLE lane rather than three quarters of it (see Notes for the
      overlap with Lane 2's `operator_decision.py`).
- [x] Keep the backend switch mid-script and re-print with the SAME renderer:
      the semantic transcript is identical, the provider session id is not.
      Both already asserted in `main`; keep both assertions.
- [x] Put at least one assertion behind every claim the new output makes. The
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
- [x] Write `tasks/20260801-154211/chat.html` beside `architecture.html`, reusing
      that file's `:root` tokens and section shape so the two read as one set.
      Content, sourced from the accepted records rather than re-derived: the
      event model; the four owned records (`conversation`, `event`, `delivery`,
      `provider_session`) and who writes each; the settled per-turn granularity
      (`tasks/20260804-115256/DECISION.md` section 1 - one event per meaningful
      thing said, because a turn-grained row cannot answer "who said this" for
      anything inside the turn); and the retention non-decision (same record,
      section on retention - no window, no compaction, the table grows without
      bound, and that is a choice with its reason).
- [x] Link `chat.html` from `packages/chat/src/scufris_chat/README.md`'s pointer
      list, which already links every record this page compiles.
- [x] Run the boundary and example gates plus `tatr check`; both gates were green
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

## Close-out

**What and why.** `examples/chat_conversation.py` now runs the WHOLE lane and
draws what it did. Three additions: `authorize` mints a decision from the
operator's message and refuses the agent's report, a `system` notice records
which event decided, and the transcript is rendered with `rich.tree.Tree` as a
causation tree with one colour per typed actor. `tasks/20260801-154211/chat.html`
is the explainer beside `architecture.html`, compiled from the accepted records
rather than re-derived, and the package README's pointer list links it.

The demo is the lane's deliverable, so the point was to make its OUTPUT
load-bearing rather than decorative. Two gates in `tests/test_examples.py` do
that: one parses the script and requires every function `scufris_chat.__all__`
exports to be called by it, and one runs it as a subprocess and reads its stdout
back - every event drawn, drawn twice by the same renderer, attributed, and
placed under the event it answers.

**Alternatives.**

- **Asserting through stdout scraping inside the example.** Rejected for the
  reason the plan gives: the recording console makes the string the script
  vouches for and the string an operator reads ONE string. `render_transcript`
  returns the text and the caller checks it before printing, so a rendering that
  drifts from the assertion is not expressible.
- **Proving the colour claim from stdout.** Not possible: `rich` emits no ANSI
  on a pipe, so the gate would see plain text either way. The checkable half is
  `set(ACTOR_STYLES) == set(ActorKind)`, asserted in the example before anything
  renders - which is also where a missing kind would otherwise surface, as a
  `KeyError` the first time one was written.
- **A third gate for the `authorize` refusal.** Not added. The refusal is
  asserted inside the demo (the message must NAME `agent:builder`), and the
  exported-function gate is what makes the call unskippable. A test asserting
  the refusal text through stdout would pin the wording of a `PermissionError`
  that `packages/chat/tests/test_chat_authority.py` already owns.

**Deviations from the plan, both deliberate.**

- The demo appends a THIRD event - a `system` notice caused by the operator's
  message - which the plan did not name. Two events make a tree with one edge
  and one child, where "under" and "after" are indistinguishable from "the only
  other line". The third gives the causation node two children, so both guide
  glyphs are drawn and the placement assertion has something to fail on. It also
  puts a third actor kind in the output.
- `TREE_GUIDES` is a tuple rather than the plan's singular `TREE_GUIDE`, since
  `rich` draws two glyphs (`├──` and `└──`) and a tree with siblings uses both.

**Difficulties and diagnosis.** The output gate's line matcher first counted
`1. opened scufris.db ...` - the demo's own step header - as a rendering of
event 1, and failed with `assert 3 == 2` on a demo that was drawing the tree
correctly. The step numbering and the event numbering are two sequences that
share a format. The fix is a fact rather than an escape: a rendered event sits
INSIDE the tree, so `_tree_lines` requires a non-zero depth, and a step header
is flush against the left margin.

**Evidence.** Both gates were red on the base for the reason claimed:
`..._calls_every_exported_function` on `authorize` and nothing else,
`..._renders_an_attributed_causation_tree` on the flat `print` output. Four
sabotages after they went green, each restored from the index:

| Removed | Fails |
|---|---|
| the report's `causation_id` | `..._renders_an_attributed_causation_tree`: "event 2 answers event 1 and should be drawn UNDER it" |
| `ActorKind.ORCHESTRATOR` from `ACTOR_STYLES` | the example exits 1; `..._renders_an_attributed_causation_tree` and `test_offline_example_runs` |
| the re-print after the backend switch | `..._renders_an_attributed_causation_tree`: "event 1 should be rendered twice by the SAME renderer" |
| the actor from the rendered label | the example itself, exit 1, naming all three events |

Green: `nix flake check` (all 6 checks - ruff, ruff-format, mypy over 250 files,
the whole pytest suite, records, filesize), `python -m pytest` in the worktree
(1138 passed, 1 pre-existing skip), and `tatr check`. The two regression suites
the Notes name, `tests/test_examples.py` and `tests/test_package_boundaries.py`,
went from 15 passed on the base to 17.

**Reflection.** The gate that reads an example's stdout is worth more than the
one that reads its exit code, and it is cheap - but only if the assertions are
STRUCTURAL. Every claim here is derived from the demo (its `append_event` calls,
its `Actor` constants, its `TREE_GUIDES`), so rewording a body does not turn it
red and dropping half of what it renders does. The one place that rule was
broken - matching a line by its number alone - is exactly where the false
positive came from, and the repair was to find the structural fact ("events are
drawn inside the tree") that the number was standing in for.

## Review round 1

Six findings, all fixed; see REVIEW.md for the per-finding responses. Four of
them were the same defect the Reflection above congratulates itself on avoiding:
an assertion whose docstring claims more structure than the assertion carries.

- The attribution check ran against the whole of stdout, and the demo names
  every actor in its step 3 and step 6 prose, so it passed on a tree carrying no
  attribution at all. It now searches only RENDERED lines. Sabotage: with the
  actor stripped from `render_transcript`'s label AND from `tree_problems`'
  author check - so the demo's own gate could not mask the miss - the assertion
  goes red on `agent:builder`; the old form was green on that same demo.
- The causation check hardcoded event 1 as everything's parent. `_causation`
  now walks the demo's `append_event` calls, binds each assigned name to its
  sequence number, and reads `causation_id=<name>.id` back to the event it
  names, returning `(3, {2: 1, 3: 1})` here. The edges are derived the way the
  count already was.
- `GUIDE_CHARACTERS` and `_depth` were re-typed in the test module that says
  re-typing the demo's tables is the one drift that keeps a gate green. Both are
  deleted; `_tree_lines` takes `demo` and uses `demo.GUIDE_CHARACTERS` and
  `demo.depth`.
- Three smaller ones: "8 checks" corrected to 6, the demo's docstring now names
  `rich` as a dev-shell dependency rather than `scufris-chat`'s, and the
  explainer's sample output marks its elision.

`nix flake check` failed once during this pass on
`tests/test_app.py::test_agent_run_reaches_done_and_persists_session` and passed
on the immediately following run. That file is not in this branch's diff and the
test is green in the dev shell; the race is real but pre-existing, and it is now
task 20260804-173304 rather than this branch's problem.

Re-verified after the fixes: `nix flake check` exit 0 (all 6 checks),
`tests/test_examples.py` green in the dev shell (11 passed), `ruff check` and
`ruff format` clean, `tatr check` exit 0.

**Round 2 fixes.** One defect in two places plus one honesty fix. The tree's
placement claim was asserted as `depth(answer) > depth(asked)` in both the gate
and the demo's own `tree_problems`, which pins "deeper than SOME ancestor"
rather than "under the event it answers" - round 1's R1.3 fixed how the edges
are DERIVED and left the predicate they feed alone, so a demo redrawing event 3
under event 2 passed both. The rule now lives once, as `parent_line` beside
`depth` in the demo, and both callers take it. Alternative considered and
rejected: comparing against a known indent width, which would have re-typed
`rich`'s layout into the assertion - the drift R1.4 removed. The discriminating
evidence is not a red run, because on this 3-event demo the incidental
`'├── '` guide check fires first: both predicates were evaluated directly
against the sabotaged run's stdout, and event 3 gives OLD True, NEW False.

Re-verified after round 2: `nix flake check` exit 0, `nix develop --command
python -m pytest` exit 0 over the whole suite, `tatr check` exit 0, and the
demo exits 0. Every sabotage applied here was reverted and `git diff` re-read
before committing.
