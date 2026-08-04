# Decision: Render agent reports as attributed quotations

- DATE: 20260804-154854
- STATUS: ACCEPTED
- TASK: 20260804-115321
- TAGS: chat, actors, authorization, v0.2.0

## Context

`tasks/20260729-220835/DECISION.md` section 3 ratified one rule: an agent report
is data, never an instruction, and only an `operator` event may satisfy a stop
gate. Two of this task's three claims are already held by landed code.
`20260804-115256` made the author a typed `Actor` with two CHECK constraints
behind it, and `20260804-115320`'s `assemble_context` attributes every LINE and
declares in its preamble that only the operator's are instructions
(`test_assembled_context_attributes_its_actors`). What is missing is the half
with teeth: nothing in the tree yet turns "this event's actor is `operator`" into
something a stop gate can be shown to require.

Nothing consumes it yet either. The flow guard is Lane 4 and the host approval
decoupling is Lane 2; both are unwritten. So the choice is not "which
authorization mechanism" but "what is the smallest artifact that makes the
refusal a property rather than a convention, without building Lane 2's task
here".

Two constraints bound it. `packages/chat/src/scufris_chat/store.py` is 582 lines
against a 600-line cap, so this lands as a new module either way - that is
forced, not chosen. And the epic (`tasks/20260801-154211/TASK.md`, Lane 2 and
Lane 4) has already ratified the SHAPE of the consumer: `advance()` takes an
`OperatorDecision` that only `chat` can mint, so an agent cannot construct the
argument.

## Decision

`packages/chat/src/scufris_chat/decisions.py` holds a frozen `OperatorDecision`
and the one function that mints it:

```python
authorize(conn, conversation_id, event_seq) -> OperatorDecision
```

Three properties, each with a test:

- It **re-reads the committed row** under the caller's open connection rather
  than accepting an `EventRecord` the caller hands over. A value passed in is a
  value the caller can build; a row read back inside the unit of work is one an
  operator actually said. This is `causing_event`'s shape, for `causing_event`'s
  reason - with no FOREIGN KEYs, the store checks what the schema will not.
- It **refuses every non-operator actor** with `PermissionError`, naming the
  actor it refused. `agent:<id>` is the case the Story is about;
  `orchestrator` and `system` are refused by the same clause, so the coordinator
  landing later inherits the refusal instead of being an unconsidered fourth
  case.
- `OperatorDecision` **cannot be constructed outside this module**. Its
  `__init__` takes a module-private witness, so the type is importable for an
  annotation - Lane 4 needs that - while `authorize` stays its only mint. Python
  cannot make this absolute; the witness plus its test is what turns "an agent
  would have to go out of its way" into something a reviewer can point at.

It lives in `packages/chat`, not `scufris_core`, until a second package consumes
it. `chat` already owns the actor, the event and the transaction rule, and it is
the only package that exists to mint from. The epic puts the value type in
`core` because `flow` and `hostctl` both consume it; neither is written, and
`CORE_MODULES` is an allowlist whose entries are meant to cost a justification.
The move to `core` is Lane 2's, at the moment its second consumer makes the
justification real.

## Alternatives considered

**A boolean predicate - `may_authorize(event) -> bool`.** Smaller, and it would
satisfy the words of the Story. It loses on the DoD's "at the type level where
possible": a caller that forgets to ask gets no error, so the refusal stays a
convention every call site has to keep - which is the exact failure
`actors.py`'s own docstring says the typed actor exists to end. It would also be
re-litigated in Lane 4, which has already ratified the capability argument.

**Define `OperatorDecision` in `scufris_core` now.** It is where the epic puts
it, so doing it here would save a later move. Rejected: it adds a module to a
deliberately expensive allowlist for a type with no cross-package consumer, and
Lane 2's task is chartered to define it alongside both subjects and both
consumers. Deferring costs one rename plus one allowlist line, in the task that
has the requirement.

**Mint from a caller-supplied `EventRecord`.** No connection needed, and it reads
more naturally. Rejected: the record is a plain frozen dataclass, so
`OperatorDecision(EventRecord(actor=Actor(ActorKind.OPERATOR), ...))` mints a
decision from an event nobody ever said. The re-read is what ties the capability
to the transcript.

**Do nothing here and let Lane 4 build it.** The rendering claims are already
green, so this task would close having changed nothing. Rejected: it is the one
claim the Story calls the security property, and Lane 4 is behind Lane 2 and the
carve. Leaving it unbuilt leaves the ratified rule with no artifact at all.

## Consequences

Easier: Lane 4's `advance()` and Lane 2's `approve()` have their argument
already minted and tested, and the refusal is one clause in one module rather
than a comparison at each call site. The coordinator landing later gets the
`orchestrator` refusal for free.

Harder, and honest:

- **The token has no production caller in this release yet.** Its only callers
  are tests until Lane 4. That is a deliberate exception to the concept budget,
  taken because the epic names the consumer's signature and the alternative is
  shipping the rule with no artifact.
- **A move is booked.** Lane 2 relocates the type to `scufris_core`. Anything
  that annotates against `scufris_chat.OperatorDecision` before then changes its
  import.
- **Who may APPEND an operator event is still unconstrained.** `append_event`
  takes the actor from its caller, so the guarantee is "only an operator EVENT
  authorizes", not "only the operator can write one". Closing that needs the
  inbound channel on `event`, which
  `packages/chat/src/scufris_chat/README.md` section 8 defers with no reader
  yet. Recorded as a limit, not solved here.
- **`PermissionError` rather than a package exception.** One raiser, and its
  builtin meaning is exact. A `NotAuthorized` type would be a concept with no
  second catcher.
