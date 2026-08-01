# Decision: the persistence mechanism, migration, and recovery policy

- DATE: 20260801-110341
- STATUS: ACCEPTED
- TASK: 20260801-100405
- TAGS: v0.2.0, reliability, storage, architecture

## Context

20260729-102146 measured the shipping stores racing (97-173 exceptions per 200
writes, 186 of 200 reasoning turns lost silently, 37 of 102 finished agents
left half-recorded) and left nine constraints plus four open questions.
`tasks/20260801-100405/SPIKE.md` measured the two candidate replacements
against those constraints, the axes this task named, and the workload
20260729-220835 will bring.

This record is what the three implementation tasks must honor. It decides the
mechanism, the boundary, the transaction API, and the migration and recovery
policy. It migrates no production state and creates no product schema.

## Decision

### 1. Mechanism

> SUPERSEDED by `tasks/20260729-102147/DECISION.md`: the library is SQLAlchemy
> 2.0, not stdlib `sqlite3`. The database, its location, its file mode and the
> pragma table below are unchanged.

One SQLite database at `<state_dir>/scufris.db`, through the stdlib `sqlite3`
module. No ORM, no migration framework, no new entry in `pyproject.toml` or
`uv.lock`.

Connection settings, applied to every connection at open:

| Pragma | Value | Why |
|-|-|-|
| `journal_mode` | `WAL` | readers never block behind writers; SPIKE.md scenario 8 measures 0.01ms vs 91ms p50 |
| `synchronous` | `FULL` | a `nixos-rebuild switch` or OOM mid-write must not lose a committed record |
| `busy_timeout` | `5000` | a contended writer waits rather than raising `SQLITE_BUSY` at a route |
| `foreign_keys` | `ON` | off by default in SQLite; the schema's references are only real with it on |

File mode `0600` on `scufris.db`, `-wal` and `-shm`: the database holds auth
session identifiers, which `scufris/auth/store.py` protects with the same mode
today.

### 2. The transaction API

One public entry point, a synchronous context manager over a unit of work:

```python
with db.transaction() as tx:      # BEGIN IMMEDIATE ... COMMIT, or ROLLBACK
    tx.execute(...)
```

Rules the implementation must enforce, each traceable to a measured failure:

1. **One connection per thread**, created lazily and kept in a
   `threading.local`. `sqlite3` reports `threadsafety == 3` in this
   environment, but a connection per thread is what makes `BEGIN IMMEDIATE`
   mean "this thread's transaction".
2. **A transaction never spans an `await`.** Every coroutine on the loop thread
   shares that thread's single connection, so an awaited transaction can
   interleave with another coroutine's transaction on the same connection and
   corrupt both. This is the one hazard the measurements do not show and the
   rule that removes it.
3. **Loop-thread callers offload**: `await asyncio.to_thread(unit_of_work)`,
   where `unit_of_work` is an ordinary synchronous function using
   `db.transaction()`. It runs on a pool thread with its own connection, so
   rule 2 holds structurally rather than by review. Measured cost: 0.28ms p50
   loop lag against 1.03ms for committing inline (SPIKE.md scenario 10). There
   is no second, async store API.
4. **`BEGIN IMMEDIATE`, not deferred.** A deferred transaction that upgrades to
   a write mid-way can fail with `SQLITE_BUSY` after doing work; an immediate
   one takes the write lock up front and waits within `busy_timeout`.
5. **The transaction is the read-modify-write boundary, not just the write.**
   Every window the predecessor listed - `mark_finished`'s `preserve_signal`
   read, `OutcomeStore.acknowledge`, `SessionRegistry.add`/`remove`,
   `SettingsStore.apply`, `DigestStore.mark_delivered`, and
   `SchedulerStore.get`, which writes on a read path
   (`scufris/scheduler.py:107`) - opens where the state is READ. A lock around
   the persist alone closes none of them.
6. **Commit-or-revert, and prefer no in-memory mirror at all.** Today a failed
   persist leaves the record live in the process (97 of 97) and the next
   successful write publishes it. Stores read through to the database. Where a
   cache is genuinely required, it is populated only after a successful commit.
7. **Damaged is not empty.** No tolerant loader. `sqlite3.DatabaseError` on
   open propagates; the app refuses to start rather than presenting an empty
   store, which is how one collision silently cost every record before.

### 3. The state boundary

One database for ALL app-owned mutable state. Two named exceptions:

| Out of the boundary | Why |
|-|-|
| `scufris/hostd/audit.py` | Written by the ROOT helper, not the app. Different privilege domain (an app-writable audit is not an audit), different process (no app-side transaction could cover it), different shape (`O_APPEND` per record, rotation as its durability policy). Nothing in the boundary reaches it. |
| Provider-owned native session transcripts | Owned by Codex/Claude/OpenCode on their own disk layout. Scufris reads them; it does not adopt them. 20260729-220835 decides what semantic record Scufris keeps above them. |

Everything the predecessor inventoried joins: projects, settings overrides,
agents, session registry, run outcomes, reasoning turns, digests, schedules,
auth sessions, and host proposals. Three of those needed an explicit answer:

**Host proposals join, as a decision journal; the helper stays authoritative
for what is pending.** `HostActionStore` is an in-memory `OrderedDict` today
(`scufris/host_actions.py:182`) and the queue is rebuilt from the root helper
after a restart (`HostApprovalService.refresh_pending`,
`scufris/host_approvals.py:287`). That recovery covers what is still PENDING
and nothing else: the decision, the operator string, the deny reason and the
apply result exist only in process memory and are lost on restart. Those rows
become durable. `refresh_pending` keeps its current additive semantics - the
helper remains the source of truth for the pending set, the database is the
app's record of what it decided and told the requesting agent. This is the
answer the epic's Done Means 4 and manual acceptance 5 were waiting on.

**The reasoning sidecar does not survive as per-session files.** It becomes
rows keyed `(session_id, seq)`. It is the one store with a genuine append
workload, which SPIKE.md scenario 3 measures as the worst case for snapshots -
append cost rising 5.47ms to 12.67ms across 5000 records while SQLite stays
flat - and it is the store the predecessor caught losing 186 of 200 turns
silently.

**Auth sessions join.** No in-process race today (it already holds a lock), but
it rewrites the whole file on every authenticated request; that becomes one
`UPDATE`. It also carries the residual exposure of a fixed temp path, which the
boundary removes. Its 0600 requirement propagates to the database file.

**The boundary spans processes.** SPIKE.md scenario 5 measured the JSON
candidate silently losing 150 of 300 writes across two processes with
`raised=0`; SQLite landed 300/300. The MCP subprocesses (`mcp_server.py:81` and
siblings) open the same database rather than their own copy of the state.
Single-writer stops being an unstated assumption.

### 4. Schema and migration policy

> PARTLY SUPERSEDED by `tasks/20260729-102147/DECISION.md`: versioning is
> Alembic, not `PRAGMA user_version`, and the legacy import is gated on a
> `legacy_import` table rather than riding the version ladder. The import
> policy table, the already-damaged-store rule, the pre-migration backup and
> the recovery diagnostics below are unchanged.

**Versioning.** `PRAGMA user_version`, one monotonically increasing integer.
Migrations are an ordered list of `(version, callable)`; each runs inside one
transaction that also bumps `user_version`, so a migration either fully applied
or never ran. No migrations table, no framework.

**Legacy JSON import.** The import rides the same `user_version` ladder, so it
arrives in the same three phases the implementation lane lands in:
20260729-102147 imports `projects.json` at v1, 20260801-100409 imports agent,
session, outcome, settings and reasoning state, 20260801-100413 imports auth,
host, schedule and digest and provides the single documented entry point for a
whole state directory. The invariant is per VERSION, not per directory: each
version's import is one transaction covering every store it claims, so a
failure anywhere in it leaves `user_version` unchanged and those legacy files
untouched, and re-running retries that version from the start. What is ruled
out is a store half-imported. A store whose version has not been reached yet
keeps reading its legacy JSON and keeps working, which is what lets each task
land green on its own. Per store, the import must:

| Requirement | Policy |
|-|-|
| Idempotency | gated on `user_version`; a second run is a no-op (measured: 25 rows after two runs, not 50) |
| Backup | copy each legacy file to `<name>.pre-sqlite.bak` before reading it |
| Legacy retention | the migration NEVER deletes a legacy JSON file; the operator does, after they are satisfied |
| Validation | parse each record through its pydantic model; a record that fails validation fails the whole import with the file and the record identified |
| Corrupt input | REFUSE with file, line, column and message (measured: `REFUSED: projects.json is damaged at line 2 col 1: Extra data`). Never import partial data, never treat damaged as empty |
| Partial recovery | one transaction per `user_version` step, covering every store that step claims; rollback restores the pre-import state exactly and the stores keep reading their legacy JSON |
| Rollback | the legacy files plus their `.bak` copies remain readable by the previous Scufris version |
| Downgrade | supported only while the legacy files are still present: an older Scufris reads them and ignores `scufris.db`. Changes made after the migration are NOT downgraded. One-way once the operator deletes the legacy files; the documentation must say so in those words |

**A store that is already damaged on an operator's machine.** Refuse to start,
name the file, and state the remedy: restore `<name>.pre-sqlite.bak`, or move
the file aside, after which the migration treats it as absent and imports the
rest. No repair heuristic, no quarantine flag, no silent skip. This is the
fourth of the predecessor's open questions.

**Backups of the database itself.** Before every schema migration,
`VACUUM INTO '<state_dir>/scufris.db.pre-v<N>.bak'` - one statement, one
consistent file, no coordination with writers.

**Recovery diagnostics.** `PRAGMA integrity_check` is exposed through an
operator-reachable check so "is my state healthy" is a command rather than a
`python -c`. Which surface is the implementer's call: `scufris/checks.py` is
today a HOST check registry driven by `HostInspector` (disk, failed units,
thermal, nix store, flake), with `check_scufris` as its one app-facing entry,
so an app-state probe is a plausible but not obvious fit there.

### 5. Constraints on the implementation lane

1. `with_suffix(".json.tmp")` does not survive anywhere in `scufris/`.
2. `mark_finished` commits the session mapping, the outcome and the agent row
   in ONE transaction (SPIKE.md scenario 2: 100/100 torn otherwise).
3. Tolerant loaders that return an empty store on damage are removed, not
   ported.
4. Tests use a file-backed database under `tmp_path` with the production
   pragmas. `:memory:` is ~100x cheaper but cannot be reopened, so every
   restart-survival proof needs the file form; the measured cost is ~10ms per
   fixture, or ~9s if all 896 collected tests took one, which the suite can
   afford. Do not diverge `synchronous` in tests to save 2-4x on a cost that
   size.
5. No conversation, activity-event or delivery tables are created by this epic.
   SPIKE.md proves the chosen store carries them - ordered append, correlation
   index, `PRIMARY KEY (channel, idempotency_key)` for idempotent delivery,
   retention by `DELETE`, and an atomic state-plus-event commit - through a
   normal `user_version` migration when 20260729-220835 designs them.

## Alternatives considered

Full measurements in `tasks/20260801-100405/SPIKE.md`.

- **Locked atomic JSON snapshots** (per-store `threading.RLock`, unique temp
  name, `fsync` of file and directory, non-tolerant loader, commit-or-revert) -
  the strongest form of the incumbent design, and the primary rejected
  alternative. It PASSES the headline test: 200/200 records, 0 exceptions under
  8 concurrent writers. Rejected on four measurements it cannot pass. Multi-file
  commits tear 100/100 when interrupted, because a lock gives mutual exclusion,
  not atomicity, and undoing already-written files means hand-writing a rollback
  journal. Reads queue behind writers at 91ms p50 / 151ms p99 on the thread that
  serves every dashboard poll. Two processes silently lose half their writes
  with `raised=0`. Append cost grows with history (5.47ms to 12.67ms over 5000
  events, 21x slower in total than SQLite), which is precisely the shape of the
  next scheduled workload and of the reasoning store that already exists. It
  also leaks orphaned temp files on every kill, with no owner to clean them up.
- **Do nothing.** 97-173 exceptions per 200 writes on the shipping code, per
  the predecessor. Not viable.
- **SQLAlchemy + Alembic.** Buys portability to a server database the
  single-host deployment model has no plan to want, at the cost of two
  dependencies in `uv.lock` and the uv2nix closure. Reversible later at low
  cost, because the schema is already SQL.
- **`aiosqlite`.** A dependency to get what `asyncio.to_thread` already gives,
  plus a second async store API alongside the sync one. Measured offload cost
  without it: 0.28ms p50 loop lag.
- **Postgres or any server database.** Contradicts the single-host deployment
  model and adds a service to `nix/`.
- **One SQLite file per store.** Discards the only reason to want a database
  here - the cross-store transaction in scenario 2 - and re-introduces manual
  lock ordering via `ATTACH`.

## Consequences

**Gained.** Multi-record commits become atomic, which is the epic's
`mark_finished` bug. Read-modify-write windows close at the transaction, not at
the persist. Reads stop queueing behind writes. The process boundary stops
being an unstated assumption. The append-only workload the next epic needs
costs a flat sub-millisecond per record instead of a growing multi-millisecond
one. Schema evolution, backups and an integrity check become one statement
each.

**Paid, all measured.**

- ~4x more disk for an append-heavy store (5.4MB vs 1.3MB over 5000 events),
  before WAL checkpointing is tuned.
- ~10ms per isolated test fixture against ~1.5ms for a JSON file, a stable 7x.
  Bounded by ~9s even if all 896 collected tests took one.
- `-wal` and `-shm` files appear beside the database and must be included in
  any operator backup advice.
- Every store's read path is rewritten, not just its write path, because
  constraint 6 removes the in-memory mirrors. This is the bulk of the work in
  20260801-100409 and 20260801-100413.
- SQL in the codebase for the first time; the transaction API and its rules
  must be documented in `scufris/README.md` before the second store migrates,
  or each store re-derives them.

**Not addressed here.** WAL checkpoint tuning under sustained event append is
deferred to 20260729-102203, where a real retention policy exists to tune it
against. `busy_timeout=5000` is a starting value; no scenario reached
contention.

**Reversal.** The mechanism is reversible until an operator deletes their
legacy JSON files: the import copies rather than moves, and the previous
version reads what it left behind. After that, reversal means writing an
export. The documentation must say this plainly.
