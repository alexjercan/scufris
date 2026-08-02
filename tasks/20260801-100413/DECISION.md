# DECISION - migrating auth, host, schedule and digest state

Four choices this task is committed to, each one a place where a reviewer would
otherwise reasonably ask "why not the other way".

## 1. The epic's headline proof MOVES rather than being duplicated

`test_concurrent_state_mutations_survive_restart` is named by the epic's Done
Means 1 and by this task's DoD, and it exists today at
`tests/test_projects.py:277` covering projects only. Two names for one claim is
how a proof rots: the epic would report green off a test that never learned
about host proposals.

So it moves to `tests/test_db_state_boundary.py` and widens there to two agent
completions plus a host proposal. `tests/test_projects.py` keeps the store-level
projects tests; the cross-store boundary proof is not one of them any more, and
was only there because projects was the first store to land.

## 2. `LoginThrottle` stays in memory

It is in `scufris/auth/store.py` next to `SessionStore` and is NOT migrated.

Failed-login timestamps are a rate-limiting window, not state an operator can
lose: a restart clearing them costs an attacker's progress, not the operator's.
Persisting it would put an unauthenticated caller's input on the write path of
the database that holds live session ids - one `BEGIN IMMEDIATE` per failed
password guess - which is a worse property than the one it would buy. The task
title names auth STATE; the sessions are that state.

## 3. Session renewal writes on every authenticated request

`SessionStore.get` renews `last_seen`, so every authenticated request becomes
one offloaded `BEGIN IMMEDIATE` plus one `UPDATE`. That is a write lock on the
request hot path, contending with agent completions.

Taken anyway, unchanged from today's semantics. The JSON store rewrote the whole
session file on the same path, so this is strictly cheaper, and the transaction
is a single keyed update inside `asyncio.to_thread`. The obvious mitigation -
skip the write unless `last_seen` has aged past some fraction of the idle window
- is deferred: it is a new knob and a new expiry edge case, and nothing measured
asks for it. If the epic's burst test shows contention, that measurement is what
justifies it, and it can be added without changing any caller.

## 4. `HostActionStore` persists the WHOLE record, proposal snapshot included

Per 20260801-100405 DECISION.md 3, host proposals join the boundary as a durable
decision journal. The alternative was persisting only the decision fields and
re-fetching the proposal from the helper on read.

Rejected: `refresh_pending` is additive and the helper expires proposals in
minutes, so a decided action's proposal is exactly the thing the helper will
soon stop returning - and "what did I approve last Tuesday" is the question the
journal exists to answer. The snapshot is stored as JSON text in one column
rather than as columns, because `ProposalView` is the helper's protocol type:
shredding it into a schema would make every helper protocol change a database
migration, and nothing queries inside it.

The pending set stays the helper's. Nothing in this task deletes a record the
helper stopped listing, and the `MAX_ACTIONS` reap keeps its rule - a decided
row goes before a pending one - now as a delete inside the insert's transaction.
