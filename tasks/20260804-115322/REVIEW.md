# Review: Prove Lane 1 with the conversation demo and the chat explainer

- TASK: 20260804-115322
- BRANCH: feature/chat-lane1-demo

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) tests/test_examples.py:317 - the attribution claim is not
  pinned to what it names. `assert label in stdout` matches anywhere in the
  demo's output, and all three labels already appear in step 3's refusal lines
  and in step 6's assembled context, so the assertion passes on a tree that
  carries no attribution at all. The docstring two screens up says "a transcript
  that lost its attribution ... would still exit 0. This reads the output an
  operator reads", and the DoD names this test as the proof the operator reads
  an attributed transcript; neither holds for attribution today. Require each
  label inside a RENDERED line: build
  `rendered = [line for seq in range(1, appended + 1) for line in _tree_lines(stdout, seq)]`
  and assert `any(label in line for line in rendered)`.
  - Response: fixed as directed. The labels are now required inside a RENDERED
    line, built from `_tree_lines` over every appended event. Sabotage: the
    actor dropped from `render_transcript`'s label AND from `tree_problems`'
    author check (so the demo's own gate could not mask it) turns the new
    assertion red with "no rendered event names agent:builder"; on the old
    `label in stdout` that same demo was green.

- [x] R1.2 (MINOR) tasks/20260804-115322/TASK.md:187 - the close-out records
  `nix flake check` as "all 8 checks"; `flake.nix:250` defines six and the run
  evaluates exactly six (`ruff`, `ruff-format`, `mypy`, `pytest`, `records`,
  `filesize`). Replace "all 8 checks" with "all 6 checks".
  - Response: fixed. The close-out now reads "all 6 checks"; the six it names
    are the six `flake.nix` defines.

- [x] R1.3 (MINOR) tests/test_examples.py:322 - the causation assertion
  hardcodes event 1 as the parent of every later event (`asked =
  _tree_lines(stdout, 1)[0]`, then `_depth(answer) > _depth(asked)` for
  `2..appended`), while the close-out claims every assertion is derived from the
  demo. This one is not: a demo whose event 3 answered event 2 would pass
  without that edge ever being checked, and a root-level fourth event would go
  red for the wrong reason. Derive the parent the way `appended` is derived -
  walk the `append_event` calls in the AST, read each `causation_id=<name>.id`
  keyword back to the event it names, and assert depth and order against that
  event rather than against event 1.
  - Response: fixed. `_causation` walks the `append_event` calls in source
    order, binds each assigned name to its sequence number, and reads every
    `causation_id=<name>.id` keyword back to the event it names; it returns
    `(3, {2: 1, 3: 1})` for today's demo, checked directly. The assertions now
    iterate the derived edges rather than `range(2, appended + 1)` against
    event 1, so an event that answers event 2, or a root-level fourth event,
    is asserted against its own parent. The test also requires `causes` to be
    non-empty, so a demo that drew no edge at all cannot pass by having
    nothing to iterate.

- [x] R1.4 (MINOR) tests/test_examples.py:197 - `GUIDE_CHARACTERS` and `_depth`
  are re-typed from `examples/chat_conversation.py:96,161` inside the very
  module whose `_load_example` docstring says re-typing the demo's tables "would
  let the assertion and the demo drift apart in the one direction that keeps
  this green", and which already imports `TREE_GUIDES` off `demo` for that
  reason. Delete lines 195-197 and 248-250, take `demo.GUIDE_CHARACTERS` and
  `demo.depth`, and pass `demo` into `_tree_lines` - its only caller has it in
  scope.
  - Response: fixed. The test module's `GUIDE_CHARACTERS` and `_depth` are
    gone; `_tree_lines` takes `demo` and uses `demo.GUIDE_CHARACTERS` and
    `demo.depth`, and the depth assertions call `demo.depth` directly.

- [x] R1.5 (NIT) examples/chat_conversation.py:44 - the demo is `scufris_chat`'s
  claimed proof in `EXAMPLES_BY_MEMBER` and its comment says only the two
  members' `src` is needed, but `rich` is declared by the root
  `pyproject.toml:20` alone, not by `packages/chat` or `packages/core`
  (`core_unit_of_work.py`'s `sqlalchemy` is a declared dep of the member it
  proves; this is not). Add a sentence to the module docstring naming `rich` as
  coming from the dev shell rather than from `scufris-chat`'s dependencies.
  - Response: fixed. The module docstring now names `rich` as the one import
    from outside the two members, declared by the root `pyproject.toml` rather
    than by `scufris-chat`.

- [x] R1.6 (NIT) tasks/20260801-154211/chat.html:994 - the sample output
  truncates event 3's body to "event 1 authorized the rebuild" while the demo
  prints "...; the agent's report did not". The refusal line two blocks up marks
  its elision with `...`; do the same here or paste the full line.
  - Response: fixed. Event 3's body in the sample now ends
    "event 1 authorized the rebuild; ...", marking the elision the way the
    refusal block above it does.

Independently re-derived rather than taken from the reviewer: R1.1, by running
the demo and reading the labels back out of steps 3 and 6, where they sit
outside the tree - so the `in stdout` test is satisfiable with no attribution
rendered at all. Note the demo's OWN author check does still catch a dropped
label (exit 1), so the behaviour is guarded; what is missing is the gate this
task added to guard it.

Checks rerun by the recording pass: `nix flake check` exit 0, six checks;
`python -m pytest` green in the worktree; `tatr check` exit 0; both `test:`
proofs pass on their stated criterion.

- Process signal: five of the six findings are about the test module rather
  than the demo, and four of those are the same shape - an assertion whose
  docstring claims more structure than the assertion carries. A step that says
  "put at least one assertion behind every claim the new output makes" does not
  by itself say "and pin each one to the substring that carries the claim".

Pending user checks, neither resolvable here: proof 3 (`manual:` - user reads
`chat.html` and agrees it explains the lane) and proof 4 (`manual:` - user runs
the demo and follows what happened from its output alone).

## Round 2

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

Every round-1 Response verified against the current tree and confirmed:
R1.1 (`rendered` is built from `_tree_lines` over every appended event, and a
dropped actor turns it red), R1.2 (TASK.md:187 reads "all 6 checks", the six
`flake.nix:250` defines), R1.3 (`_causation` walks the `append_event` calls and
returns `(3, {2: 1, 3: 1})`), R1.4 (no `GUIDE_CHARACTERS` or `_depth` remain in
the test module), R1.5 (the docstring names `rich` as the root
`pyproject.toml`'s), R1.6 (chat.html:994 marks its elision). Boxes ticked
above on that confirmation.

- [x] R2.1 (MAJOR) tests/test_examples.py:375 - the placement assertion is
  `demo.depth(answer) > demo.depth(asked)`, which pins "deeper than SOME
  ancestor" rather than "under the event it answers". The Step requires
  "assert the tree places the report under the message it answers, not merely
  that both appear", and the DoD names this test as the proof the operator
  reads causation as a tree; neither holds. R1.3 fixed the DERIVATION of the
  edges and left the predicate those edges feed unchanged. Re-derived by
  sabotage rather than taken from the reviewer: replacing
  `render_transcript`'s `parent = nodes.get(causes.get(event.event_seq, 0),
  tree)` with `nodes.get(event.event_seq - 1, tree)` draws event 3 under event
  2 though it answers event 1, and BOTH placement assertions stay green - the
  run goes red only on the incidental `'├── ' in stdout` guide check, which a
  demo with any sibling pair elsewhere would satisfy. Replace the depth
  comparison with a parent check that needs no indent constant: find the last
  rendered line before `answer` whose `demo.depth` is strictly less than
  `demo.depth(answer)`, and assert it is `asked`.
  - Response: fixed as directed, with the rule implemented ONCE. The demo grew
    `parent_line(lines, line)` beside `depth` - the nearest preceding line
    drawn shallower, stopping at the first shallower line even when it is the
    root, since walking past it would read the prose above the tree as
    ancestry - and the gate now asserts
    `demo.parent_line(stdout.splitlines(), answer) == asked`. Taken off `demo`
    for R1.4's reason rather than re-typed here. Proof that the new predicate
    pins what the old one left open, on the reviewer's own sabotage
    (`parent = nodes.get(event.event_seq - 1, tree)`, which draws event 3 under
    event 2 though it answers event 1): evaluated against that run's stdout,
    event 3 gives OLD `depth(answer) > depth(asked)` = True and NEW
    `parent_line(...) == asked` = False. The guide check at line 366 still
    fires first on this particular 3-event demo, which is why the predicates
    were compared directly rather than through the run.

- [x] R2.2 (MINOR) examples/chat_conversation.py:237 - `tree_problems` carries
  the same weakness as R2.1 (`if depth(answer) <= depth(asked)`), so the
  demo's own exit code cannot catch a child attached to the wrong ancestor
  either. Confirmed on the same sabotage: `nix develop --command python
  examples/chat_conversation.py` exits 0 while printing 1 -> 2 -> 3 as a
  chain. Apply the same nearest-shallower-preceding-line rule here and keep
  the existing message.
  - Response: fixed. `tree_problems` calls the same `parent_line`, so the
    demo's gate and the test's gate cannot drift. Its message now reads "drawn
    beside it or under something else rather than under it", which is the case
    it can now actually distinguish. On the sabotage above the demo exits 1
    with `event 3 answers event 1 and is drawn beside it or under something
    else rather than under it`; before the fix that same run exited 0.

- [x] R2.3 (MINOR) tasks/20260804-173304/TASK.md:30 - "Reproduces under the
  nix sandbox only, and not on every run" generalizes a trigger from a single
  sighting, and the sentence that follows narrows the flake to one test
  (`pytest tests/test_app.py` green in the dev shell) when the family is
  wider: the round-2 reviewer hit
  `tests/test_app.py::test_agent_fork_reverts_single_session` under a full
  `nix develop --command python -m pytest`, outside any nix sandbox, green
  again under `-p no:randomly`. Recording the flake is right; naming its
  trigger from one observation is what made the record wrong. Drop the "nix
  sandbox only" claim, name both observed tests, and record it as
  test-order dependent (`pytest-randomly`).
  - Response: fixed. The Notes now record it as test-order dependent under
    `pytest-randomly`, name both observed tests with the command each was seen
    under (`nix flake check` for the persistence test, a plain
    `nix develop --command python -m pytest` for the fork test), drop the "nix
    sandbox only" claim, and ask for a recorded `--randomly-seed` before the
    fix. Not reproduced by this pass - a full `-p no:randomly` run is green -
    which is the point of the finding rather than a rebuttal to it.

Checks rerun by this recording pass, in the worktree: `nix flake check` exit 0;
`nix develop --command python -m pytest` exit 0, one skip; `tatr -r . check`
exit 0. The sabotage above was applied and reverted; `git status` is clean.

- Process signal: round 1 named the "docstring claims more structure than the
  assertion carries" family and fixed four of its members, but R1.3 was
  directed at deriving the causation edges and stopped there. A fix aimed at a
  derivation does not automatically reach the predicate that consumes it, and
  the round-1 Response read as complete because the derivation it described
  was.
- Process signal: this branch seeded a task record for a flake it saw once and
  described the trigger as established. A seeded record is the right move; the
  claim inside it still needs the same evidence bar as the branch's own.

Pending user checks, neither resolvable here: proof 3 (`manual:` - user reads
`chat.html` and agrees it explains the lane) and proof 4 (`manual:` - user runs
the demo and follows what happened from its output alone).

## Round 3

- REVIEWER: out-of-context
- VERDICT: APPROVE

All three round-2 Responses verified against the current tree and confirmed.
R2.1: `tests/test_examples.py:375` is
`demo.parent_line(stdout.splitlines(), answer) == asked`. R2.2:
`examples/chat_conversation.py:256` calls the same helper, and the chain
sabotage now exits 1 with its message where it exited 0 before. R2.3: the
seeded record names both tests and their commands, and drops the sandbox
claim. Boxes ticked above on that confirmation.

- [ ] R3.1 (MINOR) tests/test_examples.py:343 - the assertion behind "rendered
  twice by the SAME renderer" is a COUNT of two, and the gate never compares
  the two copies: every later assertion resolves to the first
  (`_tree_lines(...)[0]` at 369-370, and `parent_line`'s `lines.index` at 375,
  which finds the first of two identical lines). A second rendering that lost
  its attribution or redrew its tree is invisible to this test. It is not
  unguarded - the demo's own `after_rendered != rendered`
  (`examples/chat_conversation.py:453`) catches it and the gate asserts the
  demo's exit code - which is why this is MINOR rather than MAJOR. Slice the
  two rendered blocks out of stdout and assert the second is byte-identical to
  the first, or run the placement and attribution checks against each
  occurrence rather than `[0]`.
  - Response:

- [ ] R3.2 (NIT) examples/chat_conversation.py:180 - `parent_line` identifies
  its target by VALUE (`lines.index(line)`), so on a list carrying duplicates
  it silently answers about the first occurrence, and raises `ValueError` for a
  line absent from `lines`. The test caller passes exactly such a list - the
  full stdout, with the transcript printed twice. Harmless today because the
  two blocks are byte-identical, so the first occurrence is the right answer;
  latent the moment they are not, which is the case R3.1 describes. Take the
  target's POSITION rather than the string, and have `_tree_lines` return
  positions.
  - Response:

Independently re-derived by this recording pass rather than taken from the
reviewer: the round-2 process signal that an isolating red run existed.
Sabotaging `render_transcript` to attach event 2 to the root while event 3
stays under event 1 - which preserves both guides, unlike the chain sabotage -
with `tree_problems`' placement arm disabled, fails at
`tests/test_examples.py:375` with "event 2 answers event 1 and should be drawn
UNDER it, not beside it or under something else". Confirmed and reverted. The
round-2 records are still accurate as written: both scope their claim to the
CHAIN sabotage and to DISCRIMINATING evidence, and this run is not
discriminating - the old `depth(answer) > depth(asked)` predicate fails it too,
since the two lines sit at equal depth. What the signal correctly names is that
the search for a red run stopped one sabotage short, not that a record
overstated.

Checks rerun by this pass, in the worktree: `nix flake check` exit 0,
`nix develop --command python -m pytest` exit 0, `tatr -r . check` exit 0, and
the demo exits 0. `git status` clean after every sabotage was reverted.

R3.1 and R3.2 are left open deliberately. Neither is a BLOCKER or a MAJOR, the
property R3.1 names is guarded by the demo's own exit code, and R3.2 is latent
rather than live; they ride the record into the retro rather than opening a
fourth round.

- Process signal: rounds 2 and 3 each found one weakness reachable from two
  callers - first the placement predicate, now the first-occurrence
  assumption. The shape is the split between "the demo asserts it" and "the
  gate counts it": a gate that counts what the demo asserts inherits none of
  the demo's precision, and its message reads as though it did.

Pending user checks, neither resolvable here: proof 3 (`manual:` - user reads
`chat.html` and agrees it explains the lane) and proof 4 (`manual:` - user runs
the demo and follows what happened from its output alone).
