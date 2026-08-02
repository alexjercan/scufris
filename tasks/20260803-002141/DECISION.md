# Decision: Move the configuration-change registry onto the database

- DATE: 20260803-011311
- STATUS: ACCEPTED
- TASK: 20260803-002141
- TAGS: storage,host,nixos,migration

## Context

The fifth and last store on the boundary, found by the discovery walk that
20260801-100413's review round 1 added rather than by anyone's inventory. The
mechanics are the same as `HostActionStore`'s cutover - a row model, one
revision, one `db.transaction()` per method - and if that were all of it this
record would not exist.

Two things about a configuration change are not like a host action, and both are
consequences of the same fact: a change is not a record the app writes once and
then decides about. It is a record a long-running BUILD writes to, repeatedly,
over minutes to hours, from a supervisor task that outlives the request that
started it.

## Decision

**1. The builder writes back through a `save` callback, not by holding the
store.** `ConfigChangeBuilder.stream` mutates the `ConfigChange` it was given -
state, `log_tail`, `error`, `toplevel`, `action_id` - and the registry sees those
mutations today only because it handed out the same object. Against a row it
would see nothing, and the `_settle` polls in `tests/test_nixos_config_change.py`
would spin until they time out.

So `stream` takes `save: Callable[[ConfigChange], None]` beside the `propose`
callback it already takes, and calls it after each transition. `app.py` passes
`config_changes.put`, whose upsert is what makes re-storing an existing change
the ordinary case rather than a special one.

The alternative was to pass the store in and have the builder call
`store.amend(change_id, **values)`, the shape `HostActionStore._amend` already
has. Rejected: the builder is deliberately store-free - it takes a runner, an
executor and a `propose` callback, and `tests/test_nixos_config_change.py`
constructs it with two of those and no app at all. Giving it a `Database`-backed
store to make it testable would mean a database in a test that fakes a nix build,
and the value it would buy - a narrower write than a whole-record upsert - is
worth nothing here, because the builder is the only writer of a building change
and there is nobody to lose a field to.

**2. A `building` row left by a restart is failed at startup, not left
building.** In memory, a crashed build simply vanished with the process, and the
next `POST /api/host/config/changes` for that repository was accepted. Persisted,
it stays `building` forever, and `building_for` then answers it: every later
build of that repository is refused with a 409, and the documented escape -
`POST /api/host/config/changes/{id}/cancel` - cannot clear it, because cancelling
requires a live supervisor run and the supervisor was restarted too. The
operator's only route out would be deleting rows from `scufris.db` by hand.

So `ConfigChangeStore.abandon_builds()` runs once at startup, beside the
`sessions.prune` sweep that is there for the same class of reason, and moves
every `building` row to `failed` with an error that says the server restarted
while the build was running. This is honest as well as convenient: the build is
in fact not running, nothing was proposed, and `FAILED` is already the terminal
state that carries no proposal. What the operator gains over the in-memory
behaviour is that the attempt is still on the list with a reason, instead of
having never existed.

The alternative of resuming the build was not seriously considered: the nix
build's output is streamed through a supervisor bus that a restart destroys,
`ConfigBuildEvent` history is in-process by construction, and nix keeps what it
already built in the store, so a re-proposed change reuses that work anyway.

## Alternatives considered

- **Leave the registry in memory and drop the boundary claim.** Rejected: the
  claim is the epic's, the discovery walk falsifies it, and the test currently
  carries a named exclusion pointing at this task.
- **A `repo` column on `config_change`, so `building_for` is one WHERE.**
  Rejected: it duplicates a field of the `resolved` JSON, and the rows it would
  scan are the `building` ones - at most a handful, filtered in Python.
- **Make `building_for`-then-`put` one transaction.** Deferred with reason: the
  race it would close is already closed by the supervisor's
  `serialize_key=f"config:{repo}"`, and the 409 is documented as the visible half
  of that guard. Tightening it is a change to the concurrency contract, not to
  where the state lives, and this task is the latter.

## Consequences

- The builder gains a second callback parameter. Its one production call site is
  `scufris/app.py`; its test call sites construct it directly and will pass a
  `save` that appends to a list or writes to a store, as they already do for
  `propose`.
- A configuration change is now durable, which means a build's `log_tail`
  (bounded at 16000 characters) and its resolved revision are on disk under the
  state directory's 0600 files rather than only in memory.
- `test_post_host_state_uses_declared_persistence_boundary` loses its exclusion
  block and quantifies over every discovered store with no exceptions, which is
  what the epic's boundary claim needs it to do.
