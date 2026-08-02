# Decision: Migrate auth, host, schedule and digest state

- DATE: 20260803-090000
- STATUS: ACCEPTED
- TASK: 20260801-100413
- TAGS: storage,auth,host,migration

## Context

This is the last cutover in the durability epic: after it, every app-owned store
is on `Database.transaction()` and nothing runtime writes a JSON sibling of the
state directory. Four choices it commits to, each one a place where a reviewer
would otherwise reasonably ask "why not the other way".

The stores it moves differ from the earlier ones in two ways that shape the
answers. `SessionStore` sits on the REQUEST hot path - every authenticated
request renews `last_seen` - and `HostActionStore` was never persisted at all: it
was an `OrderedDict` rebuilt on each boot from the root helper's queue, so this
task is not migrating its records but giving them a home for the first time.

## Decision

**1. The epic's headline proof MOVES rather than being duplicated.**
`test_concurrent_state_mutations_survive_restart` is named by the epic's Done
Means 1 and by this task's DoD, and it existed at `tests/test_projects.py:277`
covering projects only. Two names for one claim is how a proof rots: the epic
would report green off a test that never learned about host proposals. So it
moves to `tests/test_db_state_boundary.py` and widens there to two agent
completions plus a host proposal. `tests/test_projects.py` keeps the store-level
projects tests; the cross-store boundary proof is not one of them any more, and
was only there because projects was the first store to land.

**2. `LoginThrottle` stays in memory.** It is in `scufris/auth/store.py` next to
`SessionStore` and is NOT migrated. Failed-login timestamps are a rate-limiting
window, not state an operator can lose: a restart clearing them costs an
attacker's progress, not the operator's. Persisting it would put an
unauthenticated caller's input on the write path of the database that holds live
session ids - one `BEGIN IMMEDIATE` per failed password guess - which is a worse
property than the one it would buy. The task title names auth STATE; the sessions
are that state.

**3. Session renewal writes on every authenticated request.**
`SessionStore.get` renews `last_seen`, so every authenticated request becomes one
offloaded `BEGIN IMMEDIATE` plus one `UPDATE`. That is a write lock on the request
hot path, contending with agent completions. Taken anyway, unchanged from today's
semantics: the JSON store rewrote the whole session file on the same path, so this
is strictly cheaper, and the transaction is a single keyed update inside
`asyncio.to_thread`.

**4. `HostActionStore` persists the WHOLE record, proposal snapshot included.**
Per 20260801-100405 DECISION.md 3, host proposals join the boundary as a durable
decision journal, and the snapshot is stored as JSON text in one column rather
than shredded into columns: `ProposalView` is the HELPER'S protocol type, so a
schema here would make every helper protocol change a database migration, and
nothing queries inside it. The pending set stays the helper's: nothing in this
task deletes a record the helper stopped listing, and the `MAX_ACTIONS` reap keeps
its rule - a decided row goes before a pending one - now as a delete inside the
insert's transaction.

**5. One entry point reads the whole state directory.** `import_legacy_state(db,
state_dir)` replaces `import_projects` and `import_agent_state` as the single call
`open_state_database` makes. Taken during implementation (Step 10): with the
sources split across two entry points, one policy could not hold over all of them
- a damaged `projects.json` raised before the agent sources were attempted, so a
refusal in the first entry point silently deferred every source in the second.
One list, one loop, refusals collected and raised together, is what makes "a
damaged file fails startup while every other source still lands with its gate row"
true of the whole directory rather than of one half of it.

**6. A legacy source REPLACES what an earlier source migrated.** Taken in review
round 1 (R1.1). Collecting refusals means a later source runs even when an
earlier one was refused, and for exactly one pair that changes what the later
source writes: refuse `sessions.json` and `load_agents` migrates each
pre-registry `session_id` off the agent record into `agent_session` itself, then
gates `agents.json`. The repaired `sessions.json` arrives for agents that now
have rows. `load_sessions` inserting there was a `UNIQUE constraint failed` that
no retry could clear - the agents gate row means the conflicting write is never
replayed - so the documented repair path was a permanent, unbootable startup
failure. It now upserts the mapping and replaces the agent's history rows.

Replacing rather than merging is the right rule and not just the one that
compiles: `sessions.json` is the switcher's own record of what an agent has
owned, and the id on an agent record was only ever the stand-in used before the
switcher existed. Merging would leave a stale stand-in in the operator's chat
list beside the real history. The defect predates this task - master's
`import_agent_state` has the same loop, ordering and claim - but this task
rewrote that docstring, moved it to the new single entry point and widened the
policy to the whole directory, so it is fixed here rather than deferred.

## Alternatives considered

- **Persisting only the decision fields of a host action** and re-fetching the
  proposal from the helper on read. Rejected: `refresh_pending` is additive and
  the helper expires proposals in minutes, so a decided action's proposal is
  exactly the thing the helper will soon stop returning - and "what did I approve
  last Tuesday" is the question the journal exists to answer.
- **Skipping the `last_seen` write** unless it has aged past some fraction of the
  idle window. Deferred, not rejected: it is a new knob and a new expiry edge
  case, and nothing measured asks for it. If the epic's burst test shows
  contention, that measurement is what justifies it, and it can be added without
  changing any caller.
- **Keeping `import_projects`/`import_agent_state` as public entry points** beside
  the new one. Rejected as a second way to do the same thing with weaker
  guarantees; `import_legacy_file` stays as the per-source unit the loop uses.

## Consequences

- Every app-owned store constructor takes the `Database`, which
  `test_post_host_state_uses_declared_persistence_boundary` now enforces by
  DISCOVERING the stores on `app.state` rather than listing them: a new store
  wired with a path fails that test rather than eroding the boundary quietly.
  The discovery found one store the list had missed - `ConfigChangeStore` is
  still an in-memory `OrderedDict` - which is now excluded by name against
  20260803-002141 rather than silently unchecked. "Every app-owned store" is
  therefore true of the four this task names plus the three already migrated,
  with one declared exception, not of the whole app yet.
- Auth, scheduler and digest call sites in `async def` bodies must offload with
  `asyncio.to_thread`; the engine's guard turns a missed one into a loud error
  rather than a lock held under the loop.
- An operator's whole state directory is read in one call at first startup, and
  `auth_sessions.json`, `schedules.json` and `digests.json` join the existing
  backup/gate/refusal policy. Host actions have no legacy source, which the module
  docstring and `scufris/README.md` say explicitly so the absence is not read as
  an oversight.
- `scufris/db/legacy.py` became the package `scufris/db/legacy/` (`gate.py`,
  `loaders.py`) when the added loaders pushed it past the 600-line source cap.
