# A0: agent runtime foundation (de-singleton + background supervisor, no request timeout)

- STATUS: OPEN
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

- [ ] Add `scufris/eventbus.py`: an `EventBus` with a bounded ring buffer of
      `(seq, StreamEvent)` (monotonic `seq`), `publish(event)`,
      `subscribe(after_seq=0) -> AsyncIterator[tuple[int, StreamEvent]]` that
      first replays buffered events with `seq > after_seq` then streams live via
      a per-subscriber `asyncio.Queue`, and `close()` that ends all subscribers.
      Fan-out to many subscribers; a slow or dropped subscriber (bounded queue,
      drop-oldest or detach) must never block the publisher or other
      subscribers. This is the ADR-001 bus; `after_seq` is the `Last-Event-ID`
      replay hook.
- [ ] Add `scufris/supervisor.py`: a `Supervisor` managing background runs keyed
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
- [ ] Config (`scufris/config.py`): add `agent_max_concurrent: int` (default 4)
      and `agent_heartbeat_seconds: float` (stall guard, e.g. 300). Keep
      `agent_timeout_seconds` as the INTERACTIVE per-turn budget only; background
      runs pass `budget_seconds=None`. Add the new keys to `SettingsStore`
      WRITABLE_KEYS if they should be live-editable (mirror existing knobs).
- [ ] De-singleton the cwd seam: add a `cwd: str | None = None` parameter to the
      codex runner/streamer (`_stream_codex_exec` / `_stream_app_server` /
      `_run_codex_exec`) and pass `cwd=` to `create_subprocess_exec` (defaulting
      to the current behaviour when `None`). Change the `list_sessions(home,
      os.getcwd())` call site (app.py:582) to read through a small helper so a
      per-agent cwd can be supplied later. The single chat agent still uses the
      server cwd, but the seam is parameter-driven and tested with a non-default
      cwd.
- [ ] Wire the existing chat through the supervisor + bus: refactor
      `POST /api/chat/stream` so the turn runs as a supervised background job for
      the default agent and the endpoint relays that run's bus as SSE (preserve
      the leading padding comment, the SSE headers, image handling, and the
      `AgentUnavailable` error frame). Remove the request-held `chat_lock`
      (serialization is now the supervisor's `serialize_key` for the default
      agent). Keep `POST /api/chat` and `POST /api/chat/reset` correct (reset
      waits for no active run of that agent).
- [ ] Tests (`tests/test_eventbus.py`, `tests/test_supervisor.py`,
      `tests/test_app.py`): bus fan-out to N subscribers + replay after a seq +
      publisher-not-blocked-by-a-dead-subscriber; supervisor runs a job that
      sleeps past 120s without being killed (budget None), enforces the
      concurrency cap, cancels a heartbeat-stalled run, and captures an error;
      the cwd seam lists different sessions for two cwds; `POST /api/chat/stream`
      still yields `tool`/`done` frames (via the bus) AND a client that
      disconnects mid-stream does not cancel the run (it completes and is
      visible via the supervisor status).
- [ ] Full check suite green; write the close-out in this TASK.md.

## Definition of Done

- `EventBus` fans out to multiple subscribers, replays after a given seq, and a
  dead/slow subscriber never blocks the publisher
  (test: `test_eventbus.py`).
- The supervisor runs a job well past the old 120s limit without a timeout kill,
  honours `agent_max_concurrent`, and cancels a heartbeat-stalled run
  (test: `test_supervisor.py`).
- A turn runs off the request: an SSE client that disconnects mid-stream does
  not cancel the run; the run finishes and its terminal state is observable via
  the supervisor (test: `chat_stream_survives_client_disconnect`).
- The cwd seam is parameter-driven: `list_sessions` and the codex streamer honour
  a supplied cwd distinct from the server cwd
  (test: `sessions_listed_per_cwd`).
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
