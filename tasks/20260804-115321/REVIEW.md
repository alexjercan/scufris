# Review: Render agent reports as attributed quotations

- TASK: 20260804-115321
- BRANCH: feature/agent-report-quotations

## Round 1

- REVIEWER: out-of-context
- VERDICT: REQUEST_CHANGES

- [x] R1.1 (MAJOR) packages/chat/src/scufris_chat/decisions.py:69 - the witness
  does not survive `dataclasses.replace`. `__post_init__` checks only the
  IDENTITY of a shared sentinel, and `replace` copies the existing instance's
  `_witness` through, so any holder of one legitimate decision re-targets it at
  arbitrary coordinates and an arbitrary actor with a stdlib one-liner that
  names no private symbol. Re-derived independently on this worktree:
  `dataclasses.replace(real, actor=Actor(ActorKind.AGENT, "evil"),
  conversation_id="other", event_seq=999)` returns a decision. That is below
  the bar `DECISION.md` sets ("an agent would have to go out of its way") and
  contradicts the DoD ("cannot be constructed without the module-private
  witness") and `README.md`'s "A decision cannot be constructed outside the
  module". Defining `__replace__` is not the fix - it catches `copy.replace`
  but `dataclasses.replace` does not dispatch to it. Bind the witness to what
  it attests: `authorize` passes `(_WITNESS, conversation_id, event_seq,
  actor)` and `__post_init__` requires element 0 to be `_WITNESS` and the rest
  to equal the instance's own fields, so a copied witness disagrees with any
  changed field. Add the `dataclasses.replace` leg to
  `test_agent_event_cannot_satisfy_a_stop_gate` beside the two constructor
  legs.
  - Response: fixed in this commit. The witness is now
    `(_WITNESS, conversation_id, event_seq, actor)` and `__post_init__` checks
    the sentinel by identity and the remaining three elements against the
    instance's own fields, so a witness copied through `dataclasses.replace`
    agrees only with the decision it was minted for.
    `test_agent_event_cannot_satisfy_a_stop_gate` gained the re-targeting leg
    and an unchanged-copy leg beside it, so what is refused is the RE-TARGETING
    and not `replace` itself; relaxing the check back to the sentinel alone
    fails that test and no other. The README and both docstrings say what the
    binding buys.
- [x] R1.2 (MAJOR) CHANGELOG.md:10 - the diff adds a module and two public
  exports and records nothing under `[Unreleased]`. `AGENTS.md:87` requires a
  `CHANGELOG.md` entry for a notable change, and both sibling lanes of this
  epic did it (`68b6d85`, `594db53`). The existing chat bullets already narrate
  the rendering half; this is the half with teeth and it is unrecorded. Add an
  `### Added` bullet naming the operator decision, `authorize`'s `LookupError`
  and `PermissionError` refusals, and the two accepted limits - no production
  caller until the flow guard, and `append_event` still taking its actor from
  the caller.
  - Response: fixed in this commit. An `### Added` bullet under `[Unreleased]`
    names the mint, the re-read, both refusals and both accepted limits, and
    says no schema changed and nothing an operator sees.
- [x] R1.3 (NIT) packages/chat/src/scufris_chat/decisions.py:67 - `_witness` is
  in `__eq__` and `__repr__`, so the `repr()` a consumer writes to its journal
  carries `_witness=<object object at 0x...>`. Declare it
  `field(repr=False, compare=False)`. Independent of R1.1's fix.
  - Response: fixed in this commit. `repr()` is now
    `OperatorDecision(conversation_id='c1', event_seq=1, actor=Actor(...))`.
- [x] R1.4 (NIT) packages/chat/src/scufris_chat/decisions.py:109 -
  `Actor(ActorKind(row.actor_kind), row.actor_agent_id)` re-derives what
  `store.py:576`'s `_record` already derives from the same two columns. One
  line today, two places to change if the actor columns ever move. Reuse a
  shared helper rather than duplicating the widening.
  - Response: fixed in this commit. `store._actor(row)` holds the widening and
    both `_record` and `authorize` call it.
- [x] R1.5 (NIT) packages/chat/tests/test_chat_authority.py:210 -
  `transcript_lines[:-1]` drops the last line unconditionally on the comment
  that it is the epilogue. If the epilogue were removed from
  `assemble_context`, the test would silently stop checking the last real event
  line instead of going red. Assert the dropped line IS the epilogue before
  slicing.
  - Response: fixed in this commit. The test asserts the dropped line starts
    with the epilogue text before slicing it off.

Verified independently by the recording pass, not by the reviewer's report
alone: the full check suite green in the worktree (`ruff check`,
`ruff format --check`, `mypy` 250 files, `pytest` 1136 passed / 1 pre-existing
skip, `check_file_size.py`, `tatr check`); the close-out's conversation-scoping
sabotage reproduced exactly, deleting the `conversation_id` predicate failing
`test_agent_report_renders_as_attributed_quotation` and no other test; and
R1.1's bypass executed against this worktree's interpreter.

The close-out is honest. All four sabotages in its table reproduce as tabled,
the `cmd:` proof's grep is the 11 hits it claims, and the `### 3.1` placement
deviation is argued correctly - a new numbered section would renumber the
surface table and section 8, which this task's own accepted `DECISION.md` cites
by number.

- Process signal: the two MAJORs are both plan-shaped rather than
  implementation-shaped. R1.2 is a repository-wide rule the Steps did not name
  even though both sibling lanes obeyed it, and R1.1 is a property the Steps
  specified by its MECHANISM ("`__init__` takes a module-private witness")
  rather than by what it has to guarantee, so an implementation that satisfied
  the literal step still left the guarantee open.

## Round 2

- REVIEWER: out-of-context
- VERDICT: APPROVE

- [ ] R2.1 (NIT) packages/chat/src/scufris_chat/README.md:153 - the bullet
  headline "**A decision cannot be constructed outside the module.**" states an
  absolute that the same bullet then qualifies three sentences later ("Python
  cannot make this absolute"). Two routes do still mint a decision the
  transcript never backed - a hand-authored `pickle` payload whose `__reduce__`
  calls `object.__new__(OperatorDecision)`, and `object.__new__` plus
  `object.__setattr__` inline - and neither is closable in Python without a
  concept this task deferred to Lane 4. The body is accurate; only the headline
  overstates. Replace it with "**The ordinary ways to construct a decision are
  closed outside the module.**" No code change.
  - Response:

R1.1 HOLDS. The witness is `(_WITNESS, conversation_id, event_seq, actor)` and
`__post_init__` compares its tail against the instance's own fields. Relaxing
the check back to the sentinel alone fails
`test_agent_event_cannot_satisfy_a_stop_gate` and no other test. The
`dataclasses.replace` re-targeting raises `TypeError`, and the unchanged-copy
leg beside it proves the refusal is the re-targeting rather than `replace`. The
two remaining routes both bypass `__init__` outright and name the private
`_witness` field, which is the "goes out of its way" bar `DECISION.md` sets -
materially unlike Round 1's ordinary stdlib call naming nothing private. Ruled
not a finding; R2.1 is the wording that outran it.

R1.2 HOLDS - an `### Added` bullet under `[Unreleased]` carries the mint, the
re-read, both refusals and both accepted limits.
R1.3 HOLDS - `repr()` is now
`OperatorDecision(conversation_id='c1', event_seq=1, actor=Actor(...))`, and
equality still holds across two separately built witnesses.
R1.4 HOLDS - `store._actor` holds the widening; `_record` and `authorize` both
call it.
R1.5 HOLDS - the epilogue is asserted before the slice drops it.

Checks re-run whole by the recording pass on `8ca40ec`: `ruff check`,
`ruff format --check`, `mypy` (250 files), `pytest` (1136 passed, one
pre-existing skip), `check_file_size.py`, `tatr check` - all exit 0. The `cmd:`
proof's grep is 11. No `manual:` proofs.

- Process signal: the first out-of-context Round 2 reviewer completed every
  verification above and then looped for several minutes trying to coax a
  summary line out of pytest; it was stopped and its one open question - the
  two `__init__`-bypassing routes - was put to a second out-of-context reviewer
  with a narrow brief. The exception is recorded here rather than silently: no
  round-2 judgement was made inside the implementing context, but it took two
  reviewers to get one report.
