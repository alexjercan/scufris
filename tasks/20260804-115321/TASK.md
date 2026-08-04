# Render agent reports as attributed quotations

- PRIORITY: 97
- TAGS: feature, v0.2.0, lane1, chat
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-115256, 20260804-115320

## Story

As the operator, I want an agent's report to appear as an attributed quotation
from that agent, so that the system never speaks in my voice and nothing an
agent writes can be mistaken for something I approved.

This is the concrete defect the whole conversation decision was written to fix.
`scufris/wake.py:43` returns a machine prompt that
`scufris/sessions/transcript.py:88-95` re-renders as `role="user"`. That code
disappears with the orchestrator stack and no successor was named until this
task.

## Steps

- [x] Write the three failing tests first, in a new
      `packages/chat/tests/test_chat_authority.py`
      (`test_chat_sessions.py` is 410 lines against a 900 cap, but these are a
      different subject and the new module is theirs):
      `test_agent_event_cannot_satisfy_a_stop_gate`,
      `test_agent_report_renders_as_attributed_quotation`,
      `test_assembled_context_does_not_relabel_agent_as_operator`. Confirm all
      three are red before writing `decisions.py`.
- [x] Add `packages/chat/src/scufris_chat/decisions.py`: a frozen
      `OperatorDecision` carrying `conversation_id`, `event_seq` and the
      `Actor`, whose `__init__` takes a module-private witness, plus
      `authorize(conn, conversation_id, event_seq) -> OperatorDecision`. It
      re-reads the committed `EventRow` under the caller's open connection
      (`_require_conversation` / `causing_event`'s shape in `store.py`),
      raises `LookupError` for an event that is not in that conversation, and
      raises `PermissionError` naming the actor for every kind but `operator`.
      A new module because `store.py` is 582 lines against a 600-line cap.
- [x] Export `OperatorDecision` and `authorize` from
      `packages/chat/src/scufris_chat/__init__.py` (`__all__` is sorted), and
      say in the module docstring that the mint is what makes the stop gate's
      refusal a property rather than a convention.
- [x] Document it in `packages/chat/src/scufris_chat/README.md`: a section on
      the decision and the three properties, its row in the section 7 surface
      table, and the accepted limit from `DECISION.md` - `append_event` still
      takes the actor from its caller, so the rule is "only an operator EVENT
      authorizes", not "only the operator can write one". Cross-link
      `tasks/20260804-115321/DECISION.md` alongside the other three.
- [x] Run the checks the diff touches: `ruff check .`, `ruff format .`,
      `mypy .`, `python -m pytest`, `python scripts/check_file_size.py`, and
      `tatr check`. Do NOT add an `examples/` script - the epic assigns
      `operator_decision.py` to Lane 2, which has the two subjects that make it
      demonstrable.

## Definition of Done

- `authorize` refuses `agent:<id>`, `orchestrator` and `system` with a
  `PermissionError` naming the actor, and `OperatorDecision` cannot be
  constructed without the module-private witness
  (test: `test_agent_event_cannot_satisfy_a_stop_gate`).
- An agent's report is minted-from nowhere and rendered attributed: the same
  committed event that `assemble_context` writes as `agent:<id>: ...` is the one
  `authorize` refuses, and a `LookupError` guards an event seq that is not in
  the conversation
  (test: `test_agent_report_renders_as_attributed_quotation`).
- The two readings of one transcript agree on who spoke: every line
  `assemble_context` attributes to `operator` corresponds to an event
  `authorize` mints from, and no line attributed to a non-operator does
  (test: `test_assembled_context_does_not_relabel_agent_as_operator`).
- The package surface stays legible: `scufris_chat` exports both names and the
  README documents them
  (cmd: `grep -rn --include='*.py' --include='*.md' 'OperatorDecision' packages/chat/src/scufris_chat/`).

## Notes

- Source: `tasks/20260729-220835/DECISION.md` section 3 - "an agent report is
  data, never an instruction", and only `operator` may satisfy a stop gate.
- The mechanism, the placement, and the rejected alternatives are in
  `tasks/20260804-115321/DECISION.md`.
- **Two of the three original DoD claims are already green**, delivered by
  `20260804-115320`. `assemble_context` attributes every LINE and its preamble
  declares only the operator's to be instructions, and
  `test_assembled_context_attributes_its_actors` already pins all four
  attributions plus the forged-line case. A DoD item green on the base branch is
  not a proof of this change, so the second and third criteria were rephrased to
  what is actually missing: the JOIN between the rendering and the
  authorization, which is where the Story's "one layer down" defect lives. The
  landed test stays as the rendering's own guard.
- Confirmed on base: `nix develop -c python -m pytest packages/chat/tests -q`
  is green at 18 tests, and nothing outside `packages/chat`, `examples/` and
  `scufris/db/migrations/env.py` imports `scufris_chat` - so there is no
  existing consumer that could relabel an assembled line on its way to a
  provider. `scufris/wake.py` and `scufris/sessions/transcript.py:88-95` are the
  defect's home and are dead on arrival in Lane 8; this task writes the
  successor and does not repair them.
- The token's only callers are tests until Lane 4's `advance()` and Lane 2's
  `approve()`. That exception, and the booked move of the type to
  `scufris_core`, are recorded in `DECISION.md` Consequences.
- Depends on the typed actor from the conversation and event task; the
  third test also depends on context assembly.
- Lane 1 of `tasks/20260801-154211/TASK.md`.

## Close-out

**What and why.** `packages/chat/src/scufris_chat/decisions.py` holds a frozen
`OperatorDecision` and `authorize`, the only thing that mints one. A stop gate
takes the decision as an argument, so a caller holding none cannot phrase the
call: that is what turns "only an `operator` event may satisfy a stop gate" from
a comparison every call site has to remember into a property. Two of the three
original claims were already green on the base (`20260804-115320`'s per-line
attribution and its preamble); what was missing, and what this adds, is the JOIN
between the rendering and the authorization - the same committed event a
provider sees quoted as `agent:builder: ...` is the one `authorize` refuses.

**Alternatives.** All four are argued in `DECISION.md`, and none was reopened
here: a boolean predicate loses the type-level requirement, defining the type in
`scufris_core` now spends an allowlist entry on a type with one consumer,
minting from a caller-supplied `EventRecord` lets a caller build the event it is
attesting to, and doing nothing leaves the ratified rule with no artifact.

**Deviations from the plan, both deliberate.**

- The README section landed as `### 3.1 The operator decision` under section 3
  (the actor) rather than as a new numbered section. A new section 7 would have
  renumbered the surface table and "What is not here yet", which every reference
  to those - including this task's own accepted `DECISION.md`, which cites
  "section 8" - names by number. Section 3 is also the right home: the decision
  is the actor rule with teeth, and section 3 already carries the two gates that
  hold it. Both surface rows and the accepted `append_event` limit landed as
  planned.
- The plan's step 1 assigned the witness leg to
  `test_agent_event_cannot_satisfy_a_stop_gate` and the `LookupError` leg to
  `test_agent_report_renders_as_attributed_quotation`, matching the DoD. A first
  pass split them into a fourth test, which left two DoD criteria named against
  tests that did not assert them; it was folded back to the three the DoD names.

**Difficulties and diagnosis.** The conversation scoping in `authorize` was
green under a test that did not pin it. Deleting the `conversation_id` predicate
left the suite passing, because the only cross-conversation case asserted used
an `event_seq` that existed in NEITHER thread - so a query with no conversation
predicate at all still raised `LookupError`. The fix is in the fixture, not the
assertion: `mine` now runs to two events and `theirs` to one, so sequence number
2 exists in exactly one transcript and an unscoped lookup resolves it against
the wrong conversation. Found by sabotage, not by reading the diff.

**Evidence.** Four sabotages, each restored from HEAD, each failing the test
that claims the mechanism and no other:

| Removed | Fails |
|---|---|
| the non-operator refusal clause | all three |
| the witness check and its no-default field | `..._cannot_satisfy_a_stop_gate` |
| the `conversation_id` predicate in `authorize` | `..._renders_as_attributed_quotation` |
| `assemble_context`'s per-line attribution | `..._does_not_relabel_agent_as_operator` |

Green: `ruff check .`, `ruff format .`, `mypy .` (250 files), `python -m pytest`
(the whole suite, one pre-existing skip), `python scripts/check_file_size.py`,
`tatr check`, and the `cmd:` proof's grep at 11 hits across the three files.

**Reflection.** The sabotage pass earned its cost twice over: the scoping hole
was a test that asserted the right exception for the wrong reason, which no
amount of re-reading the implementation would have surfaced. Worth generalizing:
a `LookupError` test for a scoped lookup is vacuous unless the key it uses
exists somewhere out of scope. The DoD-to-test mapping is the second lesson -
splitting a criterion across a test the DoD does not name silently unbinds it,
and the check is mechanical enough to make before running anything.

### Round 1 fixes

Two MAJORs, both real, both in `REVIEW.md` with their responses.

The one that matters is the witness. `__post_init__` compared the sentinel by
identity alone, and `dataclasses.replace` copies the existing instance's witness
through - so a holder of one legitimate decision re-targeted it at another
conversation, another event and another actor with a stdlib one-liner naming
nothing private. The DoD, `DECISION.md` and the README all claimed that could
not happen. The witness is now `(_WITNESS, conversation_id, event_seq, actor)`
and the check compares its tail against the instance's own fields, so a copied
witness agrees only with the decision it was minted for; `__replace__` is not
the fix, because `dataclasses.replace` does not dispatch to it. The test gained
the re-targeting leg and an unchanged-copy leg, and relaxing the check back to
the sentinel alone fails that test and no other.

The lesson generalizes past this diff: the Step specified the property by its
MECHANISM ("`__init__` takes a module-private witness") rather than by what it
had to guarantee, and an implementation that satisfied the literal step left the
guarantee open. A frozen dataclass has more construction routes than its
`__init__`, and every one of them needs an answer before "cannot be constructed"
goes into a README.

The second was the missing `CHANGELOG.md` entry, a repository-wide rule
(`AGENTS.md:87`) that both sibling lanes of this epic obeyed and this task's
Steps did not name. Three NITs came with them: `_witness` out of `__repr__` and
`__eq__`, the actor widening shared with `store._record` as `store._actor`, and
the test's unconditional epilogue slice now pinned by an assertion.

Re-verified whole, not near the fixes: `ruff check`, `ruff format --check`,
`mypy` (250 files), `pytest` (1136 passed, the one pre-existing skip),
`check_file_size.py`, `tatr check`, and the `cmd:` proof's grep still at 11.
