# Spike: choose the persistence mechanism, migration, and recovery policy

- DATE: 20260801-110341
- STATUS: RECOMMENDED
- TAGS: spike, v0.2.0, reliability, storage

## Question

Which persistence mechanism should every app-owned mutable store share, what
transaction API can a synchronous FastAPI route and an asyncio callback both
use safely, and what migration and recovery policy carries an existing state
directory onto it?

The predecessor (20260729-102146) proved the CURRENT stores race and left nine
constraints. This spike does not re-prove that. It compares the two candidate
REPLACEMENTS and argues against measurements, not against a remembered picture
of either.

## Context

### The two candidates, both built as well as they can be

The comparison is only honest if the JSON candidate is the STRONG form. The
predecessor's control already showed the weak form (fixed temp name, no lock)
is unsafe, so re-measuring it would prove nothing. Both candidates live in
`tasks/20260801-100405/bench_persistence.py`:

| | `JsonStore` | `SqliteStore` |
|-|-|-|
| Layout | one file per store | one database file |
| Exclusion | per-store `threading.RLock` held across read-modify-write | `BEGIN IMMEDIATE`, WAL |
| Temp path | UNIQUE per pid+thread (predecessor constraint 2) | n/a |
| Durability | `fsync` of the temp file AND the parent directory, then `os.replace` | `synchronous=FULL`, `busy_timeout=5000` |
| Damage | loader RAISES; no tolerant empty (constraint 6) | `sqlite3.DatabaseError` |
| Failure | in-memory state reverted on a failed persist (constraint 4) | `ROLLBACK` |
| Dependency | none | none - stdlib `sqlite3`, no ORM, no `uv.lock` change |

Environment: python 3.14.6, sqlite 3.53.1 (`nix develop`), Linux x86_64,
24 cores. Reproduce any block below with

```sh
nix develop --command python tasks/20260801-100405/bench_persistence.py <scenario>
```

Scenarios: `race multi events crash procs migrate isolation readers leftovers
asyncio`; no argument runs all ten. Exit code is 0 whenever the run completes -
this is a measurement harness, not a test.

### What the deployment actually is

- One uvicorn process, no workers (`scufris/app.py:3739`); state defaults to
  `~/.local/state/scufris` (`scufris/config.py:52`), a local filesystem.
- Synchronous `def` routes run in anyio's thread pool - real OS-thread
  parallelism. Supervisor callbacks, scheduler ticks, Telegram handlers and
  host-approval hooks run on the loop thread. (Predecessor mutator matrix.)
- Other PROCESSES already open the stores: `scufris/mcp_server.py:81` builds
  its own `AgentStore`. Read-only today.

## Options considered

### The axes the task named, measured

**1. Concurrent writes into one store** (`race`, 8 threads x 25 creates)

```text
  locked JSON  expected=200 after_restart=200 raised=0 wall=0.998s per_write=4.99ms
  sqlite       expected=200 after_restart=200 raised=0 wall=0.362s per_write=1.81ms
```

Both are correct. This is the honest result and it matters: the headline
failure in the predecessor's record is fixable without changing mechanism. A
per-store lock plus a unique temp name is enough for ONE store in ONE process.
Everything that follows is why that is not enough.

**2. Multi-record commits** (`multi`, 200 terminal states, failure injected
before the third record)

```text
  locked JSON  interrupted=100 agents=200 sessions=200 outcomes=100 torn(session_no_outcome)=100
  sqlite       interrupted=100 agents=100 sessions=100 outcomes=100 torn(session_no_outcome)=0
```

This is `AgentStore.mark_finished` (`scufris/agent_store/store.py:502-506`),
which writes the session registry, the outcome and the agent row as three
independent files. The predecessor observed 37 of 102 finished agents ending
with a session and no outcome. Under the JSON candidate every interrupted
commit still tears - 100/100 - because a lock provides mutual exclusion, not
atomicity: the earlier records are already on disk when the failure arrives,
and undoing them means hand-writing a rollback journal, which is a database.
SQLite tears 0/100 for free.

**3. async / thread interaction** (`asyncio`, loop lag while 200 commits run
against a 3000-row store)

```text
  locked JSON  on the loop thread lag p50=13.86ms p99=19.84ms max=19.84ms
  locked JSON  to_thread          lag p50=0.38ms  p99=5.89ms  max=9.90ms
  sqlite       on the loop thread lag p50=1.03ms  p99=1.61ms  max=1.61ms
  sqlite       to_thread          lag p50=0.28ms  p99=1.07ms  max=1.07ms
```

and (`readers`, read latency while 8 writers hammer the same store)

```text
  locked JSON  reads p50=91.04ms p99=151.00ms max=155.22ms
  sqlite       reads p50=0.01ms  p99=0.17ms   max=0.84ms
```

The reader number is the decisive one. A JSON reader must take the same lock
the writers hold or it can observe a row set mid-mutation, so every read queues
behind the write storm: 91ms at the median. Those reads are the dashboard poll
and the Telegram render, and on the loop thread they are 91ms of lag for every
open tab. WAL readers do not take the write lock at all.

Both candidates need `asyncio.to_thread` for loop-thread callers; JSON needs it
badly (13.9ms -> 0.4ms), SQLite needs it for uniformity (1.0ms -> 0.3ms). Note
the hazard this creates that neither number shows: thread-local connections
mean every coroutine on the loop thread shares ONE connection, so a transaction
that spans an `await` can interleave with another coroutine's transaction on
that same connection. The rule that removes it is in DECISION.md.

**4. Migrations** (`migrate`, legacy `projects.json` -> sqlite)

```text
  first run   : imported 25 row(s), backup at projects.json.pre-sqlite.bak
  second run  : skipped (already at user_version=1)
  rows after two runs: 25 (25 means the import did not double)
  backup kept : True
  damaged file: REFUSED: projects.json is damaged at line 2 col 1: Extra data
  damaged legacy still present: True
```

`PRAGMA user_version` is a one-integer version gate that makes an import
idempotent without a migrations table or a dependency, and each version's
import fits in one transaction, so a store half-imported is not a state that
can exist. JSON has no equivalent: its "migration" is whatever shape the loader tolerates,
which is exactly the tolerance that turns damage into silent total loss.

**5. Backups**

SQLite: `VACUUM INTO 'scufris.db.pre-v3.bak'` is one statement producing a
single consistent file. JSON: copying ten files while writers run gives ten
snapshots from ten different instants; a consistent copy needs a global lock
held across all of them, which is scenario 3's reader latency applied to the
entire state directory.

**6. Observability**

SQLite ships `PRAGMA integrity_check`, `EXPLAIN QUERY PLAN`, and an `sqlite3`
shell an operator already has for reading state during an incident. The JSON
candidate's equivalent of "is my state healthy" is `python -c 'json.load(...)'`
per file.

**7. pytest isolation** (`isolation`, 200 fresh empty stores)

```text
  locked JSON      1.548ms per store
  sqlite file      10.394ms per store (tmp_path, synchronous=FULL)
  sqlite :memory:  0.149ms per store
    synchronous=FULL    10.050ms
    synchronous=NORMAL   4.475ms
    synchronous=OFF      2.339ms
```

The one axis JSON wins. Three consecutive runs gave 10.5 / 10.4 / 10.1ms for
the file form and 1.6 / 1.5 / 1.4ms for JSON, so the ~7x ratio is stable.
`:memory:` is ~70x cheaper than the file form but cannot be reopened, so every
restart-survival proof needs the file form regardless. Relaxing `synchronous`
buys about 2x (FULL 10.1 -> NORMAL 4.5) to 4x (OFF 2.3).

What that costs the suite: `pytest --collect-only` reports 896 tests. If EVERY
one took a fresh file-backed database the setup bill would be ~9s; only tests
that actually take a store fixture pay it, so the real figure is a fraction of
that. Seconds, either way - which is why constraint 4 in DECISION.md keeps the
production pragmas in tests rather than trading fidelity for 2-4x on a cost
this size.

### Beyond the named axes

**8. Two PROCESSES writing at once** (`procs`, 2 x 150 writes)

```text
  json         raised=0 on_disk=150/300
  sqlite       raised=0 on_disk=300/300
```

The worst result in this document. `threading.RLock` does not span processes,
so the second process's 150 records are gone - and `raised=0`: nothing failed,
no request errored, no log line. Total silent loss of half the writes. Fixing
it means an `flock` around every read-modify-write in every store, which is
re-implementing what SQLite's locking already does. Today nothing but the MCP
subprocess opens these files and it only reads, so this is a latent failure,
not a live one - but "single writer" is currently an accident of the process
layout, not an invariant anything enforces, and the predecessor listed it as an
open question. SQLite answers it by making it a non-question.

**9. Crash mid-write** (`crash`, SIGKILL a writer, 8 trials each)

```text
  json         readable_after_kill=8/8 records=[22, 20, 18, 22, 19, 23, 21, 19]
  sqlite       readable_after_kill=8/8 records=[27, 53, 41, 63, 54, 63, 53, 48]
```

A tie on the property that matters: neither leaves an unreadable store. This
closes the predecessor's "no crash injection" limitation. The record counts
differ only because SQLite completed more writes in the same 50ms window.

**10. What a kill leaves behind** (`leftovers`, 6 SIGKILLs each)

```text
  json         stray files=4 suffixes=[json.<pid>.<tid>.tmp x4]
  sqlite       stray files=12 suffixes=['db-shm', 'db-wal']
```

The unique temp name that fixes the race also means nothing ever cleans it up:
4 orphaned temp files from 6 kills, accumulating in the state directory forever
with no owner. `-wal` and `-shm` are recovered and reused on the next open.

### The workload from 20260729-220835

Not speculative - it is the next scheduled consumer, and the task requires the
choice to carry it without pre-creating its schema.

**Ordered append-only events, pagination, retention** (`events`, 5000 events)

```text
  locked JSON  wall=48.84s bytes=1,329,000 append_first100=5.47ms append_last100=12.67ms (x2.3) page50=6.82ms retain=5.26ms
  sqlite       wall=2.27s  bytes=5,426,704 append_first100=0.81ms append_last100=0.45ms  (x0.6) page50=0.54ms retain=10.56ms
```

A snapshot store re-serializes the entire log on every append, so append cost
grows with history: 5.47ms -> 12.67ms across one 5000-event run, 21x slower
than SQLite in total, and still climbing at the end. SQLite's append cost is
flat. This is not a hypothetical shape - `ReasoningStore.append`
(`scufris/reasoning_store.py:82-86`) is already exactly this store, already
load-append-rewrite, and already the only one the predecessor caught losing
data silently (186 of 200 turns).

One number in that block goes the other way and is stable: SQLite is 4x LARGER
on disk (5.4MB vs 1.3MB) - page overhead, an index, and an un-checkpointed WAL.
Recorded as a real cost; on a single-host dashboard it does not decide
anything.

Retention is the one axis with NO stable winner, and the quoted run is the
reason to say so rather than to claim one. Four runs of this scenario gave
JSON 5.26 / 5.19 / 4.22 / 63.95ms against SQLite 10.56 / 4.91 / 16.08 / 4.66ms
- the orderings disagree and the spread swamps the difference. Retention also
runs on a schedule rather than per request. Treat it as a wash.

Take the same care with the append figures. The block above is one run; three
more gave wall times of 46.5s, 73.8s and 98.8s for JSON against 2.3s, 2.7s and
4.2s for SQLite, with the JSON growth factor ranging x2.3 (quoted) to x140. The
quoted run is the WEAKEST demonstration of the growth effect, so the direction
transfers and the magnitude does not. As the predecessor put it: compare
verdicts across machines, not counts.

**Correlation IDs and idempotent delivery**

```text
  sqlite       duplicate delivery rejected by UNIQUE: 1/1
```

`PRIMARY KEY (channel, idempotency_key)` makes "deliver once to web and once to
Telegram" a constraint violation the database detects, not a read-then-write
the application has to serialize. `CREATE INDEX events_by_correlation` makes
"everything for this run" a single indexed scan. The JSON candidate's version
of both is a linear scan of the whole decoded log under the global lock.

**Atomic state-change-plus-event commit**

Scenario 2 is this test with different table names: a state row and its event
either both land or neither does. JSON tears 100/100; SQLite tears 0/100.

### Options rejected without measurement

- **Do nothing.** The predecessor measured 97-173 exceptions per 200 writes on
  the shipping code. Not viable.
- **Per-store lock + unique temp, keep JSON** (the JSON candidate above). The
  measured survivor of scenario 1, rejected on 2, 3, 8 and the whole workload
  block. Kept in the harness precisely so this record can say what it does and
  does not fix.
- **SQLAlchemy / Alembic.** A real ORM and migration tool, and a real
  dependency addition to `uv.lock` and the uv2nix closure. What it buys over
  `PRAGMA user_version` plus hand-written SQL is portability to a server
  database Scufris has no plan to want (`scufris/README.md`, and 20260729-102147
  fixes the single-host deployment model in its own notes). YAGNI; reversible
  later if a second backend ever appears, since the schema would already be SQL.
- **`aiosqlite`.** Wraps sqlite3 in a thread per connection. `asyncio.to_thread`
  around a synchronous unit of work is the same thing without the dependency,
  and it keeps ONE store API for both caller shapes rather than a sync and an
  async one. Scenario 10 measures the offload cost at 0.28ms p50.
- **A server database (Postgres).** Contradicts the single-host deployment
  model, adds a service to `nix/`, and buys nothing measured here.
- **Per-store SQLite files.** Keeps a database per store and loses the only
  reason to want one: scenario 2's cross-store transaction. Cross-file `ATTACH`
  transactions exist but re-introduce lock ordering by hand.

## Recommendation

RECOMMENDED: one SQLite database, `<state_dir>/scufris.db`, stdlib `sqlite3`,
WAL, `synchronous=FULL`, `busy_timeout`, `BEGIN IMMEDIATE`, one connection per
thread, no ORM and no new dependency. The full architecture, the transaction
API, the store boundary with its named exceptions, and the migration and
recovery policy are in `tasks/20260801-100405/DECISION.md`.

The short form of why, in the order the evidence lands:

1. A lock fixes the loud failure (scenario 1) and nothing else.
2. Multi-record commits tear 100/100 under any lock-based JSON design
   (scenario 2). That is the epic's `mark_finished` bug, and it is not fixable
   without hand-writing a rollback journal.
3. Reads queue behind writes at 91ms p50 (scenario 8), on the thread that
   serves every dashboard poll and Telegram render.
4. Two processes silently lose half their writes (scenario 5), with `raised=0`.
5. The next scheduled workload is an append-only event log - the exact shape
   whose cost grows with history under snapshots (scenario 3) - and it needs
   atomic state-plus-event commits (scenario 2) plus idempotency keys the
   database can enforce.

Costs accepted, all measured: 4x disk for the event workload, ~10ms per
isolated test fixture against ~1.5ms, and `-wal`/`-shm` files beside the
database. Retention cost is a wash across runs, not a win for either.

## Open questions

The predecessor's four open questions are answered in DECISION.md rather than
left open: the boundary spans processes; the reasoning sidecar becomes a table
rather than per-session files; auth sessions join the boundary; a damaged store
is refused loudly with a named remedy rather than repaired or silently emptied.

Genuinely still open, none blocking the implementation lane:

- WAL checkpointing under sustained event append is untuned.
  `wal_autocheckpoint` defaults to 1000 pages; scenario 3 never checkpointed,
  which is part of why its on-disk figure is 4x. Revisit when the activity-event
  task (20260729-102203) has a real retention policy, not before.
- `busy_timeout=5000` was never driven to contention; no scenario produced a
  `SQLITE_BUSY`. The thing to watch after 20260729-102147 lands is whether any
  transaction ever waits at all, not whether 5000 is the right number.
- Scenario 8's write counts are not comparable between candidates (sampling
  wall time differs by three orders of magnitude). Only the read latencies are.
- `procs` uses `fork` from a benchmark, not two real Scufris processes; the MCP
  subprocess is spawned differently. The file-locking semantics it exercises are
  the same, the deployment shape is not.

## Next steps

No new tasks are seeded. The three implementation tasks already exist and this
spike's job was to constrain them; their Steps were re-checked against the
decision and refined where the mechanism changed them.

- 20260729-102147 - persistence core plus the pilot store. Owns
  `scufris/state/`, the transaction API, and the fixture shape.
- 20260801-100409 - agent, session, outcome, settings, reasoning state. Carries
  the reasoning-sidecar shape change out of DECISION.md.
- 20260801-100413 - auth, host, schedule, digest, plus the whole-directory
  legacy import. Carries the host-proposal answer and the migration
  documentation.
- The epic's Done Means 4 now has its host-proposal answer: the app's proposal
  decision journal joins the boundary; the root helper stays authoritative for
  what is still pending. See DECISION.md.
