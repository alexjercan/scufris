# A0: agent runtime foundation (de-singleton + background supervisor, no request timeout)

- STATUS: CLOSED
- PRIORITY: 30
- TAGS: spike,agents,refactor

## Goal

Foundation / gating refactor for the multi-agent orchestrator. Two things
together (spike revision 1, decisions 1-2):

- **De-singleton.** Session listing hard-filters `cwd == os.getcwd()`
  (sessions.py:255-259); make it per-agent cwd. Turns inherit `os.getcwd()`;
  pass the agent's project cwd to the subprocess (`-C`/cwd) instead.
- **Background execution + event bus (ADR-001).** The global
  `chat_lock = asyncio.Lock()` (app.py:303) runs every turn inside the held HTTP
  request. Replace it with an in-process supervisor that runs agent subprocesses
  as background jobs, with a concurrency cap acting as the queue (agents past the
  cap wait). Split the run from its stream: `POST .../run` enqueues and returns a
  run id immediately; each worker publishes normalized `StreamEvent`s to a
  per-agent event bus (fan-out ring buffer, backed by the durable rollout/session
  log). `GET .../events` is a thin SSE subscriber that replays the buffer and
  streams live (drop-safe, `Last-Event-ID`); `GET /api/agents` polls coarse
  status. Replace the 120s `agent_timeout_seconds` hard kill with a per-agent
  budget + liveness/heartbeat that only catches a genuinely stuck subprocess.

SSE is kept (delivery), workers give concurrency + no timeout (execution) - the
two are orthogonal (ADR-001). Gates A1-A5. No new user-facing feature alone.

## Steps

Scope note: there is no `AgentStore` yet (that is A1), so A0 builds the runtime
components standalone and proves them by routing the EXISTING single chat agent
through them. Generic per-agent `run`/`events` endpoints land with the agent
record in A3/A4; A0 delivers the machinery + the cwd seam + the de-timeouted
execution model, tested with the `mock` backend (no codex login needed).

- [x] Add `scufris/eventbus.py`: an `EventBus` with a bounded ring buffer of
      `(seq, StreamEvent)` (monotonic `seq`), `publish(event)`,
      `subscribe(after_seq=0) -> AsyncIterator[tuple[int, StreamEvent]]` that
      first replays buffered events with `seq > after_seq` then streams live via
      a per-subscriber `asyncio.Queue`, and `close()` that ends all subscribers.
      Fan-out to many subscribers; a slow or dropped subscriber (bounded queue,
      drop-oldest or detach) must never block the publisher or other
      subscribers. This is the ADR-001 bus; `after_seq` is the `Last-Event-ID`
      replay hook.
- [x] Add `scufris/supervisor.py`: a `Supervisor` managing background runs keyed
      by `run_id`. `start(run_id, make_stream, *, serialize_key, budget_seconds,
      heartbeat_seconds)` schedules an asyncio background task under an
      `asyncio.Semaphore(max_concurrent)` (the queue); the task drains
      `make_stream()` (an `AsyncIterator[StreamEvent]`), publishing each event to
      that run's `EventBus` and refreshing a heartbeat; it records a `RunState`
      (`queued|running|done|error`, `started_at`, `last_event_at`, `error`) and
      publishes a terminal `done`/`error`. `serialize_key` makes turns of the
      SAME agent run one-at-a-time while DIFFERENT agents run concurrently.
      `budget_seconds=None` means no wall-clock cap (background runs); the
      heartbeat guard cancels only a genuinely stalled run. Expose `bus(run_id)`,
      `status(run_id)`, `list_runs()`.
- [x] Config (`scufris/config.py`): add `agent_max_concurrent: int` (default 4)
      and `agent_heartbeat_seconds: float` (default 600). Keep
      `agent_timeout_seconds` as the INTERACTIVE per-turn budget only; background
      runs pass `budget_seconds=None`. DECIDED not live-editable for A0: kept
      out of `WRITABLE_KEYS` (they are startup/env config; the
      `WRITABLE_KEYS <-> AgentConfigUpdate` mirror test would otherwise fail, and
      making them live pulls in settings-UI scope A0 does not own).
- [x] De-singleton the cwd seam: added a keyword-only `cwd: str | None = None` to
      `_run_codex_exec` / `_stream_codex_exec` / `_stream_app_server`, forwarded
      to `create_subprocess_exec(cwd=cwd)` (default `None` = inherit, unchanged).
      Keyword-only so the three stay assignable to the `StreamRunner` alias and
      no fake needs touching (avoids the protocol-signature blast radius). The
      session-listing half was ALREADY per-cwd: `list_sessions(home, cwd)` takes
      a cwd and filters on it (proven by the existing
      `test_list_sessions_filters_by_cwd_and_originator`); the app call site
      keeps `os.getcwd()` as the single-agent default until A1 supplies a
      per-agent `project_cwd` there (and wires `CodexCliAgent` to pass `cwd` -
      the one-pass change the fakes need, done with the record that needs it).
- [x] Wire the existing chat through the supervisor + bus: refactor
      `POST /api/chat/stream` so the turn runs as a supervised background job for
      the default agent and the endpoint relays that run's bus as SSE (preserve
      the leading padding comment, the SSE headers, image handling, and the
      `AgentUnavailable` error frame). Remove the request-held `chat_lock`
      (serialization is now the supervisor's `serialize_key` for the default
      agent). Keep `POST /api/chat` and `POST /api/chat/reset` correct (reset
      waits for no active run of that agent). Chat mutations reuse the same
      "chat" serialize lock via `supervisor.serialize("chat")`, so
      reset/new/switch/fork/delete still cannot interleave with a turn.
- [x] Tests (`tests/test_eventbus.py`, `tests/test_supervisor.py`,
      `tests/test_agent.py`): bus fan-out + replay-after-seq +
      publisher-not-blocked-by-a-full-subscriber; supervisor runs a slow job with
      no budget (no timeout kill), enforces the concurrency cap, serializes
      same-key turns, cancels a heartbeat-stalled run AND an over-budget run;
      the subprocess runs in the supplied cwd; existing `test_chat_stream_*`
      still pass through the bus.
- [x] Full check suite green; close-out written below.

## Definition of Done

- `EventBus` fans out to multiple subscribers, replays after a given seq, and a
  dead/slow subscriber never blocks the publisher
  (test: `test_eventbus.py`).
- The supervisor runs a job well past the old 120s limit without a timeout kill,
  honours `agent_max_concurrent`, and cancels a heartbeat-stalled run
  (test: `test_no_wall_clock_timeout_without_a_budget`,
  `test_concurrency_cap_queues_extra_runs`, `test_heartbeat_cancels_a_stalled_run`).
- A turn runs off the request: a subscriber that disconnects mid-stream does
  not cancel the run; the run finishes and its terminal state is observable via
  the supervisor (test: `test_run_survives_subscriber_disconnect`).
- The cwd seam is parameter-driven: the codex subprocess runs in a supplied cwd
  distinct from the server cwd, and session listing already filters per cwd
  (test: `test_run_codex_exec_runs_in_the_given_cwd`,
  `test_list_sessions_filters_by_cwd_and_originator`).
- `POST /api/chat/stream` still streams a turn end to end via the bus
  (test: existing chat-stream test, adapted, passes).
- The full suite passes (cmd: `nix develop --command bash -c "ruff check . &&
  mypy . && pytest -q"`), and `npm run ci` in web/ is unaffected.

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (decisions 1-2, ADR-001; "singleton /
  one-cwd" blocker; execution model).
- No external broker (no Redis/Celery) - lightweight in-process supervisor.
- The `mock` backend (`SCUFRIS_AGENT_BACKEND=mock`, agent.py:187) drives the
  supervisor in tests without a codex login.
- Generic per-agent `POST /api/agents/{id}/run` + `GET /api/agents/{id}/events`
  endpoints are deliberately deferred to A3/A4 (they need the A1 record); A0
  proves the machinery through the existing chat endpoint.
- Durable restart-replay of the bus from the rollout/session log is future work;
  A0's buffer is in-memory.

## Close-out

What changed:
- New `scufris/eventbus.py` (`EventBus`: monotonic seq, bounded replay buffer,
  fan-out, `subscribe(after_seq)` = replay-then-live, `close()`; publish is sync
  and non-blocking via drop-oldest so a dead subscriber never stalls the bus).
- New `scufris/supervisor.py` (`Supervisor`: background runs keyed by run_id,
  `asyncio.Semaphore` concurrency cap, per-serialize_key lock so same-agent turns
  don't overlap while different agents run in parallel, per-event heartbeat guard
  + optional total budget replacing the request timeout, `RunState` snapshots).
- `scufris/app.py`: chat turns now run under the supervisor and the
  `/api/chat/stream` endpoint RELAYS the run's event bus instead of iterating the
  agent inline. The global `chat_lock` is gone; `supervisor.serialize("chat")`
  provides the same serialization to both turns and the session-mutation
  endpoints. `Last-Event-ID` is honoured for reconnect replay. Supervisor is
  torn down via a `lifespan` handler (replacing a deprecated `on_event`).
- `scufris/agent.py`: `cwd` keyword forwarded to the three codex subprocess
  spawns. `scufris/config.py`: `agent_max_concurrent`, `agent_heartbeat_seconds`.

Why / alternatives:
- Execution and delivery decoupled per ADR-001: the run lives in the supervisor,
  the SSE request is a mere subscriber. This INVERTS the old
  `sse-streaming-from-a-subprocess-in-fastapi` pattern (which killed the proc on
  client disconnect) - now a disconnect is a no-op on the run, which was the
  whole point.
- Kept `cwd` keyword-only on the module runners rather than threading it through
  the `Agent` protocol / `StreamRunner` alias, to avoid the
  `protocol-signature-change-hits-the-doubles` blast radius; `CodexCliAgent` will
  pass a per-agent cwd in A1 when the record carries `project_cwd`.
- Left the two new knobs out of `WRITABLE_KEYS` (startup config) so the
  `WRITABLE_KEYS <-> AgentConfigUpdate` mirror stays intact and no settings-UI
  work is dragged into A0.

Difficulties:
- The `WRITABLE_KEYS` mirror test (`test_writable_keys_match_the_api_update_model`)
  red-flagged the new knobs immediately - good guardrail; resolved by scoping them
  as startup config.
- `on_event("shutdown")` is deprecated in this FastAPI and spammed 152 warnings;
  switched to a `lifespan` async context manager.
- mypy flagged unpacking `object` off the subscriber queue; fixed with a `cast`
  after the `_CLOSE` sentinel check.

Result: 216 backend tests pass (12 new: 5 eventbus + 7 supervisor, +1 cwd),
ruff + mypy clean, `npm run ci` green (frontend untouched).

Self-reflection: the plan's "list_sessions via a helper" step was slightly
redundant - the function was already cwd-parameterized, so the real net-new
surface was just the subprocess cwd. Reading the signature first (I did) kept
that from becoming a hollow indirection. Next time, verify a "de-singleton" claim
against the actual function signatures during planning, not at work time.
