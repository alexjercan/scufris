# Spike: inventory app-owned mutable state and reproduce the write races

- DATE: 20260801-101245
- STATUS: RECOMMENDED
- TAGS: spike, v0.2.0, reliability, storage

## Question

What mutable state does Scufris own today, who writes each store, and which of
those writers can collide? Evidence only: the mechanism, migration and recovery
decision is the successor spike (20260801-100405), which must argue against
these measurements rather than against a remembered picture of the code.

## Context

Every app-owned store repeats one write discipline, copied store to store and
stated as a virtue in most of their docstrings ("atomic write, tolerant load"):

```python
tmp = self._path.with_suffix(".json.tmp")   # FIXED path, shared by every writer
tmp.write_text(json.dumps(payload))         # not atomic; flushes in 8 KiB chunks
os.replace(tmp, self._path)                 # atomic, but of a file anyone may hold
```

The `os.replace` is atomic. Nothing else is. The temp path is derived from the
target path alone, so every concurrent writer of a store picks the SAME temp
file, and no store except `auth/store.py` holds a lock across the sequence.

### Inventory

One row per app-owned mutable store. "Gated" is whether writes are refused when
`settings_writable` is false.

| Store | Module | On-disk path | Write pattern | Record shape | Gated |
|-|-|-|-|-|-|
| Projects | `scufris/projects.py:111` | `<state_dir>/projects.json` | full-snapshot rewrite, shared tmp | JSON list of `{id,cwd,name,language,description}` | yes |
| Settings overrides | `scufris/settings_store.py:173` | `<state_dir>/settings.json` | full-snapshot rewrite, shared tmp | `{overrides: {key: value}}` | yes |
| Agents | `scufris/agent_store/store.py:142` | `<state_dir>/agents.json` | full-snapshot rewrite, shared tmp | JSON list of `AgentRecord` | CRUD only |
| Session registry | `scufris/agent_store/registry.py:80` | `<state_dir>/sessions.json` | full-snapshot rewrite, shared tmp | `{agent_id: {backend, session_id, sessions[], parent_*}}` | no |
| Run outcomes | `scufris/agent_store/outcomes.py:69` | `<state_dir>/outcomes.json` | full-snapshot rewrite, shared tmp | `{agent_id: RunOutcome}` | no |
| Reasoning sidecar | `scufris/reasoning_store.py:114` | `<state_dir>/reasoning/<session_id>.json` | load-append-rewrite, shared tmp, **errors swallowed** | `{turns: [{answer, reasoning}]}` | no |
| Digests | `scufris/digest.py:190` | `<state_dir>/digests.json` | full-snapshot rewrite of a bounded deque, shared tmp | `{digests: [Digest]}` | no |
| Schedules | `scufris/scheduler.py:101` | `<state_dir>/schedules.json` | full-snapshot rewrite, shared tmp | `{schedules: {name: ScheduleState}}` | no |
| Auth sessions | `scufris/auth/store.py:66` | `<state_dir>/auth_sessions.json` | full-snapshot rewrite, shared tmp, 0600, **under a `threading.Lock`** | `{sessions: {sid: {csrf, created_at, last_seen}}}` | no |
| Host proposals | `scufris/host_actions.py:182` | **none - in-memory `OrderedDict`** | n/a | `HostActionRecord` | n/a |

Two rows are not what the epic's Done Means assumes, and both matter to the
successor:

- **Host proposals are not persisted at all.** `HostActionStore` is a bounded
  in-memory `OrderedDict`; the queue is rebuilt after a restart by asking the
  root helper (`HostApprovalService.refresh_pending`,
  `scufris/host_approvals.py:287`). The decision, the reason, the operator
  string and the apply result live only in that process's memory, so a restart
  mid-apply loses the app's side of the record even though the helper keeps the
  proposal. Whether this store JOINS the persistence boundary or stays
  helper-derived is a decision the successor must make explicitly.
- **Auth sessions are already lock-protected.** `SessionStore` holds
  `self._lock` across read-modify-`_flush` in every mutator
  (`scufris/auth/store.py:88,109,129,148,155`), so it has no in-process race.
  It still shares the fixed temp path, so it is only safe as long as exactly one
  process writes it, and it rewrites the whole file on every authenticated
  request (`get` renews `last_seen` and flushes, `scufris/auth/store.py:139`).

### External boundary: the privileged audit log

`scufris/hostd/audit.py:170` is deliberately outside all of the above and must
stay there. It is written by the ROOT helper process, not the app: `os.open`
with `O_APPEND | O_CREAT` at 0600, one JSON line per record, appended rather
than rewritten. Three reasons it cannot join an app-side transactional store:

1. Different privilege domain. The app runs unprivileged; the log records what
   root did. An app-writable audit is not an audit.
2. Different process. Nothing in the app holds a handle to it, so an app-side
   lock or transaction could not cover it.
3. Different shape. `O_APPEND` writes are atomic per record under the size
   limits in play, and rotation (`_rotate_if_needed`) is the durability policy.
   It has no read-modify-write window to protect.

### Mutator matrix

Scufris runs one asyncio event loop plus FastAPI's anyio thread-pool. Route
handlers in `scufris/app.py` are synchronous `def`, so FastAPI dispatches them
into that thread-pool - they run in PARALLEL OS threads, on all cores.
Everything else - supervisor completion callbacks, scheduler ticks, Telegram
handlers, host-approval hooks - runs on the loop thread.

The Telegram bot is not a separate writer class: it drives the SAME
`_launch_agent_turn` path the dashboard uses (`app.py:2825,2415`) and decides
host actions through the same `HostApprovalService`
(`telegram/approvals.py:216,310`). Its writes therefore land in the loop-thread
column below, and they overlap a dashboard request exactly as a supervisor
callback does.

| Store | Thread-pool (sync routes) | Event-loop thread | Other processes |
|-|-|-|-|
| projects.json | `create/update/delete_project` (`app.py:1893,1986,2004`), `create_new_project` (`app.py:1941`) | - | MCP subprocess, read-only (`mcp_server.py:81`) |
| agents.json | `create/update/delete_agent` (`app.py:2029,2116,2177`) | `mark_running` / `mark_finished` from the supervisor completion callback (`app.py:2344,2316`); a telegram-driven turn takes the same path | MCP subprocess, read-only |
| sessions.json | `update_agent` (backend switch clears the mapping), session fork/new-chat routes (`app.py:3475,3494,3656`) | `record_running_session` mid-stream (`app.py:2257`), `mark_finished` (`app.py:2316`), telegram new-chat (`app.py:914,964`) | MCP subprocess, read-only |
| outcomes.json | `agent_request_input`, `agent_report_back`, `agent_acknowledge` (`app.py:3004,3031,3058`) - called by LIVE sub-agents over HTTP | `mark_finished` (`app.py:2316`), host-approval hooks marking an agent BLOCKED (telegram or dashboard, `host_approvals.py:210`) | - |
| settings.json | `patch_agent_config` (`app.py:1871`), the settings routes | - | - |
| reasoning/*.json | - | `reasoning_store.append` inside the turn stream (`app.py:2278`) | - |
| digests.json | the run-now button | scheduler tick -> `digests.add` / `mark_delivered` (`app.py:2672,2674,2679`) | - |
| schedules.json | the run-now button | scheduler tick -> `_store.save` (`scheduler.py:262,278,291,332`), and `_store.get` PERSISTS on a read (`scheduler.py:107`) | - |
| auth_sessions.json | every authenticated request renews `last_seen` and flushes | - | - |

Pairs that can overlap - write the same file at the same instant:

- **thread-pool x thread-pool** - all of the above. Two dashboard tabs, two
  API clients, or one client with parallel requests. This is genuine OS-thread
  parallelism, not cooperative interleaving.
- **thread-pool x loop thread** - `agents.json`, `sessions.json` and
  `outcomes.json`. The named case in the epic: an agent finishing
  (`mark_finished`, loop) while the operator edits or deletes an agent
  (`update_agent`/`delete_agent`, thread-pool). Also a live sub-agent posting
  `report_back` (thread-pool) while the same agent's turn completes (loop).
- **loop thread x loop thread** - two agents completing together, a telegram
  approval landing during a supervisor callback, or a scheduler tick landing on
  a digest write. These cannot tear a file TODAY, but only because `_persist`
  happens to contain no `await`. That is an accident of the current code, not a
  property anything enforces: one `await` added inside a persist path turns
  every loop-thread pair into the thread-pool case.
- **process x process** - not currently a write race. The MCP subprocess builds
  its own `AgentStore` (`mcp_server.py:81`) but only reads. It is still exposed
  to a torn file: the tolerant loaders return an EMPTY store rather than
  raising, so the orchestrator's "what are my agents doing" would answer
  "nothing" instead of failing.

## Reproduction

Committed script: `tasks/20260729-102146/repro_state_races.py`. It drives the
REAL store classes from a `ThreadPoolExecutor`, which is exactly the shape
FastAPI produces for a synchronous route handler.

```sh
nix develop --command python tasks/20260729-102146/repro_state_races.py
nix develop --command python tasks/20260729-102146/repro_state_races.py --writers 16 --rounds 40
```

Exit code 0 means a failure was reproduced. Defaults: 8 threads x 25 writes =
200 records per store, each record padded to 4 KiB so one snapshot spans several
`write(2)` calls - the same condition a store with real content reaches on its
own.

### Observed, run of 2026-08-01 (8 threads x 25 writes)

```text
--- projects.json ---
  expected: 200   in_memory: 200   on_disk: 200   after_restart: 200
  exceptions raised: 90
  FAILURE: a write raised

--- agents.json + outcomes.json + sessions.json ---
  expected: 200   in_memory: 200   after_restart: 200
  outcomes_after_restart: 67
  agents.json: 200 record(s), parses
  outcomes.json: 67 record(s), parses
  sessions.json: 102 record(s), parses
  exceptions raised: 174
  FAILURE: a write raised

--- unique tmp name (control) ---
  expected: 200   on_disk: 200   exceptions raised: 0
  published_regressions: 26
  worst_regression_records: 7

--- reasoning/<session>.json (errors swallowed by design) ---
  expected: 200   after_restart: 5
  exceptions raised: 0
  FAILURE: 195 record(s) unrecoverable after a restart
```

The traceback, verbatim, from `projects.json`:

```text
  File "scufris/projects.py", line 160, in create
    self._persist()
  File "scufris/projects.py", line 116, in _persist
    os.replace(tmp, self._path)
FileNotFoundError: [Errno 2] No such file or directory:
  '.../projects.json.tmp' -> '.../projects.json'
```

And, from the reasoning sidecar's swallowed-error log line:

```text
reasoning sidecar: cannot read .../shared-session.json:
  Extra data: line 8 column 2 (char 4186)
```

### What each observation shows

1. **`FileNotFoundError` from `os.replace` - 90/200 and 174/200 writes.** Writer
   B's `os.replace` consumed the shared temp file while writer A was still
   holding it; A's own `os.replace` then found nothing to rename. In the
   dashboard this surfaces as a 500 on `POST /api/projects` or
   `POST /api/agents`, and in the run engine as a terminal state that never
   reaches disk. `AgentStore.mark_finished` writes THREE files in sequence
   (`outcomes.json`, `sessions.json`, `agents.json`; `store.py:501-506`) with no
   transaction across them - which is why the run above kept only 67 outcomes
   and 102 sessions for 200 agents: the raise landed partway through and the
   record set is now internally inconsistent, not merely short.

2. **A published file that does not parse.** `Extra data: line 8 column 2` is
   the shared-temp failure at its worst. Writer B truncates and rewrites the
   shared temp while A holds an open fd on the same inode; B's `os.replace`
   publishes it as the store; A's remaining buffered chunks then land in what is
   now the LIVE file, past the end of B's valid JSON. Every loader in this
   inventory is tolerant, so the next read discards the ENTIRE file and returns
   an empty store. One collision can cost every record, silently.

3. **The failure is silent where the store swallows it.**
   `ReasoningStore._persist` catches `OSError` (`reasoning_store.py:120`), and
   the `os.replace` failure IS an `OSError`. 195 of 200 turns were lost with no
   exception, no failed request and no difference in any API response - only a
   warning log nobody is watching. Its per-session file makes the collision
   RARER in production (only same-session turns collide) but not impossible, and
   its `_load`-append-`_persist` cycle (`reasoning_store.py:82-86`) is a genuine
   read-modify-write, unlike the snapshot stores.

4. **A unique temp name is not the fix - the control proves it.** With only the
   temp path made per-writer, the run raised nothing and ended complete, but the
   published file REGRESSED 26 times, by up to 7 records. Each regression is a
   writer publishing a snapshot older than one already on disk: kill the process
   at that instant - `nixos-rebuild switch`, an OOM, a crash - and those records
   are gone. The final file was whole only because the writers share one
   in-memory dict and whoever replaced last happened to publish everything. That
   is luck, not a property, and it disappears the moment two processes write.

### Limitations

Stated so the successor does not over-read this.

- Threads, not real HTTP. The script calls the store classes directly rather
  than driving `TestClient`, so it proves the STORES race, not that any specific
  endpoint pair has been observed racing in a running server. The mutator matrix
  above is derived by reading call sites, not by instrumenting a live process.
- Single process. Multi-process writing is not exercised, because nothing writes
  these files from a second process today. The control scenario is the argument
  that it would be unsafe if anything did.
- The loop-thread x loop-thread pair is argued, not measured: it cannot tear a
  file while `_persist` stays `await`-free, and the script does not try to
  demonstrate a failure that today's code does not have.
- No crash injection. Regression 4 shows the window; it does not kill a process
  inside it. "Records would be lost on a crash here" is an inference from the
  regression count, and a sound one, but it is an inference.
- `_UniqueTmpStore` is a stand-in with the same discipline, not one of the real
  stores - it exists to isolate one variable.
- Frequencies are machine- and load-dependent. The counts above are one run on
  one machine; re-running gives different numbers with the same verdicts
  (a second run gave 93 and 175 exceptions).

## Options considered

No mechanism is chosen here; that is 20260801-100405. What this spike DID choose
between:

- **Read the code and describe the races** - rejected. The epic exists because
  an audit described them; a description is what the successor would then argue
  against. It would also have missed the two facts that changed the picture:
  host proposals are not persisted at all, and auth sessions already hold a lock.
- **Reproduce through the HTTP surface with `TestClient`** - rejected for this
  spike. It proves the same thing more slowly, and couples the evidence to route
  wiring that lanes C and D are about to move. Worth doing once as the epic's
  acceptance test (`test_concurrent_state_mutations_survive_restart`), where
  route-level truth is the point.
- **Reproduce against the store classes from threads** - taken. It is the
  smallest thing that can fail, it names the module in the traceback, and it
  survives the router extraction.
- **Add the unique-temp control** - taken, and it earned its place: it is the
  only reason this record can say that renaming the temp file is not the fix.

## Recommendation

The evidence supports these constraints on the successor's decision. They are
requirements the mechanism must meet, not a mechanism.

1. **Serialize writers per store.** The dominant failure is two writers in one
   store's persist path. Any mechanism has to exclude them; a per-store lock is
   the floor, and the floor is not currently there.
2. **Never share a temp path.** Even under a lock, a fixed temp name is only
   safe while exactly one process writes. Auth sessions are the live example of
   correct locking with this residual exposure.
3. **Make the multi-file update one unit.** `mark_finished` writing three files
   with no transaction is how the run above ended with 200 agents, 102 sessions
   and 67 outcomes. Whatever is chosen must let a run's terminal state land or
   not land as a whole.
4. **Stop treating an unreadable store as an empty one.** The tolerant loaders
   turn corruption into silent total data loss. Recovery policy has to
   distinguish "absent" from "damaged", and damaged has to be loud.
5. **Decide the host proposal store explicitly.** It has no file today. Either
   it joins the boundary or the helper stays its source of truth, and the epic's
   Done Means 4 needs whichever answer in writing.
6. **Keep the privileged audit outside.** Different process, different
   privilege, already append-only. Nothing in the boundary should reach it.
7. **Test at the level the failure lives.** The default 8x25 run reproduces in
   under a second; the epic's acceptance test can afford to be real.

## Open questions

Left for 20260801-100405, none blocking this record:

- Does the boundary span processes, or is single-writer an invariant the
  deployment enforces? The MCP subprocess reads today; nothing stops a future
  one writing.
- Does the reasoning sidecar's per-session layout survive the decision, or does
  its growing per-turn list want a different shape from the small snapshot rows?
- Is the auth session store in scope? It has no in-process race, but it rewrites
  the whole file on every authenticated request, which is a cost question rather
  than a correctness one.
- What does recovery mean for a store that is already damaged on an operator's
  machine today - repair, quarantine, or refuse to boot?

## Next steps

No new tasks are seeded: 20260801-100405 already exists as the successor and is
this spike's only consumer. Its Steps are written there, against this evidence.

- 20260801-100405 - choose the persistence mechanism, migration and recovery
  policy. Must argue against the Recommendation constraints above and answer the
  Open questions.
- The epic's Done Means 4 needs the host-proposal answer from that task; this
  record supplies the fact that the store is currently memory-only.
