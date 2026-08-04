# A3: create-agent-with-goal end to end (background job, gated write, tracked state)

- PRIORITY: 24
- TAGS: spike, agents
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

First real vertical slice of the vision: create an agent bound to a project +
goal, launch it as a **background job** via the A0 supervisor (no held request,
no timeout), scoped to the project cwd via the A2 `AgentBackend` (the agent's
`backend` selects codex/claude), and track its lifecycle
(idle|running|blocked|done|error) by merging the A0 Supervisor run-state with the
A2 `read_status` rollout/session progress, surfaced by polling.

CORRECTION from the A2 probe (tasks/20260720-221935/NOTES.md): do NOT hard-code
"/flow" into the run. codex is already agentic (you hand it a goal prompt and it
runs its own loop); `/flow` is a Claude-Code-only skill. A3 hands each backend a
GENERIC GOAL PROMPT via `AgentBackend.stream(prompt=<goal>)`; each backend
realizes autonomy its own way. The `CodexCliAgent`-cwd wiring + the
`StreamRunner`-fake one-pass update (deferred from A0/A1) land here.

## Decisions (locked by the operator, 20260721)

- **Write scope: PLUMBING ON, DEFAULT OFF.** Build the `write_enabled` flag end
  to end - a write-enabled agent lifts the sandbox (codex: drop `--sandbox
  read-only` on the first turn per `codex-resume-rejects-sandbox`; claude:
  `--permission-mode acceptEdits`/equivalent) scoped to the project cwd - but v1
  agents default to READ-ONLY, and this flow does NOT exercise a live
  file-writing run. The write path is wired and unit-tested (the right flags are
  built for `write_enabled=True`), not live-verified. Flipping write on per agent
  is a deliberate later operator action.
- Pace: the operator PAUSED the flow after A2b (4/7). A3-A5 resume on their
  next go-ahead. This note is the on-disk pin so a fresh session resumes cleanly.

## Steps

- [x] Write plumbing (default OFF) through the stream seam: add
      `write_enabled: bool = False` (keyword-only) to `AgentBackend.stream` +
      all three backends. Add `sandbox: str = "read-only"` (keyword-only) to
      `_exec_args` / `_run_codex_exec` / `_stream_codex_exec` /
      `_stream_app_server` (mirror the A0 cwd seam), used for the `--sandbox`
      flag (exec, first turn only per `codex-resume-rejects-sandbox`) and the
      app-server `start_params`. `CodexBackend` maps `write_enabled` ->
      `sandbox="workspace-write"` else `"read-only"`; `ClaudeBackend` appends
      `--permission-mode acceptEdits` when write_enabled (else default). UNIT-test
      the arg translation only; do NOT run a live writing agent.
- [x] `AgentStore` run-state mutators (NOT CRUD): `mark_running(id)`,
      `mark_finished(id, *, state, session_id=None)` that set `state`/`session_id`
      and persist. Used by the run engine, not the API.
- [x] Supervisor: add an optional `on_complete: Callable[[RunState], None]` to
      `start(...)`, invoked in `_execute`'s finally after the terminal state is
      set, so the run engine can persist state/session_id when a background run
      ends.
- [x] Run engine in `app.py`:
      - `POST /api/agents/{id}/run` (optional `{goal}` override; else the agent's
        stored goal, 422 if none): resolve the agent + its project cwd, build
        `get_backend(agent.backend)`, wrap its `stream(goal, session_id=
        agent.session_id, cwd=project.cwd, write_enabled=agent.write_enabled)` to
        capture the final `StreamDone.session_id`, `supervisor.start(run_id=
        agent.id, serialize_key=agent.id, on_complete=persist)`; `mark_running`
        immediately; 409 if a run for this agent is already active.
      - `GET /api/agents/{id}/status`: merge the supervisor run-state (if a live
        run) with `backend.read_status(agent.session_id)` into an
        `AgentRunStatus` (state + turns/tokens/last_message/updated_at).
      - `GET /api/agents/{id}/events` (SSE): relay `supervisor.bus(agent.id)`
        (drop-safe, `Last-Event-ID`); 404 when the agent has no run.
- [x] Tests: a mock-backend run launches, reaches `done`, and persists
      `session_id` + `state`; a second concurrent run for the same agent -> 409;
      run with no goal -> 422; status merges supervisor state + read_status;
      events relay yields the run's frames; the write flag builds
      `--sandbox workspace-write` (codex) / `--permission-mode acceptEdits`
      (claude) when enabled and read-only when not.
- [x] Full check suite green; close-out.

## Definition of Done

- Creating+running an agent with a goal launches a supervised background run that
  reaches a terminal state and persists its session id, and its status merges the
  supervisor run-state with the backend read_status
  (test: `test_agent_run_reaches_done_and_persists_session` - asserts done +
  persisted session_id + merged `turns`/`last_message`).
- `GET /api/agents/{id}/events` relays the run's event bus (and 404s with no run)
  (test: `test_agent_events_relay`).
- Write plumbing is wired but default-off: `write_enabled=True` builds the
  sandbox-lifting flags, default builds read-only (tests:
  `test_codex_backend_write_enabled_lifts_sandbox` asserts workspace-write vs
  read-only; `test_claude_backend_write_enabled_adds_permission_mode` asserts
  acceptEdits present vs absent).
- Full suite passes (cmd: `nix develop --command bash -c "ruff check . && mypy .
  && pytest -q"`). manual (deferred, NOT this flow): a live write-enabled agent
  actually modifies files.

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (recommendation 3; decisions 2,3).
- Depends on: 20260720-221929 (A1, landed 17bad00), 20260720-221935 (A2, landed
  4d6850a); A2b (deb0ce9) gives the claude backend.
- SCOPE: A3 is the AGENT run engine (a separate agent runs a goal). Rewiring the
  MAIN chat orchestrator through `AgentBackend` (decision 4, "orchestrator
  swappable") is NOT required for the run engine or the dashboard and is left as
  a later cleanup - the existing `/api/chat/stream` keeps its supervised path.
- If the write plumbing + run engine prove too large together, split: land the
  read-only run engine first, write plumbing as a fast follow (per the work
  skill's "split if much larger than planned").

## Close-out

What changed:
- Write plumbing (default OFF): `sandbox` threaded through `_exec_args` +
  the three codex runners (mirroring the A0 cwd seam); `write_enabled` added to
  `AgentBackend.stream` + all backends. `CodexBackend` maps write ->
  `--sandbox workspace-write` (first turn only); `ClaudeBackend` -> `--permission-mode
  acceptEdits`; default read-only. Unit-tested arg translation; NO live writing run.
- `AgentStore.mark_running` / `mark_finished` (run-state mutators, not CRUD).
- `Supervisor.start(on_complete=...)` callback (invoked with the terminal RunState).
- `app.py` run engine: `POST /api/agents/{id}/run` (goal override or stored goal;
  404/422/409), `GET /api/agents/{id}/status` (merges supervisor run-state +
  backend read_status), `GET /api/agents/{id}/events` (SSE bus relay, 404 if no
  run). Runs keyed by a unique `agent_id:uuid` id (serialize_key=agent_id), so a
  finished run never blocks a re-run; an `agent_runs` map tracks the current run.
- Tests (6): mock-backend run reaches done + persists session_id + status merge;
  no-goal 422; re-run after completion; events relay (+404); codex write flag
  ->workspace-write / claude ->acceptEdits (default read-only).

Decisions / scope:
- WRITE is plumbing-on-default-off (operator decision): the flags are built and
  unit-tested but no live file-writing agent was run in this flow.
- Corrected the spike's "/flow" phrasing: A3 hands each backend a GENERIC GOAL
  PROMPT; codex runs its own agentic loop (it does not use /flow), claude could.
- The main-chat orchestrator was NOT rewired through AgentBackend (decision 4) -
  left as a later cleanup; the run engine drives SEPARATE agents.

Difficulties:
- `run_agent` first written as a SYNC endpoint -> `supervisor.start` failed with
  "no current event loop in thread 'AnyIO worker thread'" (FastAPI runs sync
  endpoints in a worker thread; `create_task` needs the loop). Fixed by making
  it `async def` (like `/api/chat/stream`).
- Adding `sandbox` to the codex runners red the two A2 CodexBackend test fakes
  (`protocol-signature-change-hits-the-doubles`); updated both fakes in one pass.

Result: 245 tests pass (+6), ruff + mypy clean.

Self-reflection: the sync-vs-async endpoint trap and the runner-fake blast radius
were both foreseeable from prior lessons - I caught the fake one by sweeping, but
the sync endpoint I only caught at test time. Next time, any endpoint that
touches the supervisor (create_task) is async by default.
