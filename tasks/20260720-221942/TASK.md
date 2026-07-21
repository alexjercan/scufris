# A3: create-agent-with-goal end to end (background job, gated write, tracked state)

- STATUS: OPEN
- PRIORITY: 24
- TAGS: spike,agents

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

- [ ] Write plumbing (default OFF) through the stream seam: add
      `write_enabled: bool = False` (keyword-only) to `AgentBackend.stream` +
      all three backends. Add `sandbox: str = "read-only"` (keyword-only) to
      `_exec_args` / `_run_codex_exec` / `_stream_codex_exec` /
      `_stream_app_server` (mirror the A0 cwd seam), used for the `--sandbox`
      flag (exec, first turn only per `codex-resume-rejects-sandbox`) and the
      app-server `start_params`. `CodexBackend` maps `write_enabled` ->
      `sandbox="workspace-write"` else `"read-only"`; `ClaudeBackend` appends
      `--permission-mode acceptEdits` when write_enabled (else default). UNIT-test
      the arg translation only; do NOT run a live writing agent.
- [ ] `AgentStore` run-state mutators (NOT CRUD): `mark_running(id)`,
      `mark_finished(id, *, state, session_id=None)` that set `state`/`session_id`
      and persist. Used by the run engine, not the API.
- [ ] Supervisor: add an optional `on_complete: Callable[[RunState], None]` to
      `start(...)`, invoked in `_execute`'s finally after the terminal state is
      set, so the run engine can persist state/session_id when a background run
      ends.
- [ ] Run engine in `app.py`:
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
- [ ] Tests: a mock-backend run launches, reaches `done`, and persists
      `session_id` + `state`; a second concurrent run for the same agent -> 409;
      run with no goal -> 422; status merges supervisor state + read_status;
      events relay yields the run's frames; the write flag builds
      `--sandbox workspace-write` (codex) / `--permission-mode acceptEdits`
      (claude) when enabled and read-only when not.
- [ ] Full check suite green; close-out.

## Definition of Done

- Creating+running an agent with a goal launches a supervised background run that
  reaches a terminal state and persists its session id
  (test: `agent_run_reaches_done_and_persists_session`).
- `GET /api/agents/{id}/status` returns the merged run-state + progress
  (test: `agent_status_merges_supervisor_and_read_status`).
- `GET /api/agents/{id}/events` relays the run's event bus
  (test: `agent_events_relay`).
- Write plumbing is wired but default-off: `write_enabled=True` builds the
  sandbox-lifting flags, default builds read-only
  (test: `write_enabled_builds_workspace_write`, `default_run_is_read_only`).
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
