# A2: AgentBackend interface + codex runner + read-only status + unattended probe

- STATUS: CLOSED
- PRIORITY: 26
- TAGS: spike,agents,backend

## Goal

The common backend seam plus the codex implementation and the load-bearing
probe (spike revision 1, decisions 1,4):

- **`AgentBackend` interface**: `run` / `stream` / `status` / `resume`, designed
  so the store, supervisor, dashboard and orchestrator never branch on backend.
  It must hide output format, session resume, MCP config, and permission/sandbox
  model differences (see spike decision 1).
- **codex runner** behind the interface (reuses the existing app_server/exec
  machinery).
- **read-only `agent_status`** `-> {state, last_activity, current_tool, turns,
  tokens, updated_at}` from the agent's codex rollout (reuse sessions.py).
- The **orchestrator** (main chat) also routes through this interface (decision
  4 - the main agent is itself backend-swappable).
- **Probe**: run one long autonomous `codex exec` turn that invokes /flow on a
  scratch project; record unattended behaviour (approval mode, memory growth,
  liveness, failure modes) before A3 commits the UI. Resolves the open question.

## Steps

- [x] Add `scufris/backends.py`: `BackendStatus` (pydantic:
      `{session_id, state?, turns, tool_calls, input_tokens, output_tokens,
      context_window, last_message, updated_at}`) and an `AgentBackend` protocol:
      `name: str`; `stream(settings, prompt, *, session_id, cwd, image_paths)
      -> AsyncIterator[StreamEvent]`; `read_status(settings, session_id)
      -> BackendStatus | None`. `resume` is implicit (pass `session_id` to
      `stream`); the interface hides output-format/session/MCP differences so the
      store/supervisor/dashboard/orchestrator never branch on backend.
- [x] `CodexBackend(mode)` (mode `"exec"` | `"app_server"`, `name = mode`):
      `stream` delegates to `_stream_codex_exec` / `_stream_app_server` from
      `agent.py`, forwarding `cwd=` (the A0 seam) and `session_id` as the codex
      thread id; `read_status` reads the rollout via `sessions.read_context`
      (turns/tools/tokens/window) + last assistant message via
      `sessions.read_transcript`, `updated_at` via a new
      `sessions.rollout_mtime`. Returns `None` when the session has no rollout.
- [x] `MockBackend` (`name="mock"`): `stream` yields canned events (reuse the
      `MockAgent` shape); `read_status` returns a canned `BackendStatus`. A
      `get_backend(name) -> AgentBackend` factory maps
      `exec`/`app_server`/`mock` (A2b adds `claude`; unknown -> ValueError).
- [x] Add `sessions.rollout_mtime(codex_home, session_id) -> float | None`
      (public, reuses `_find_rollout`) for `updated_at`.
- [x] Tests `tests/test_backends.py`: CodexBackend.stream over the fake-codex
      script (cwd honored + a resumed session passes the thread id);
      CodexBackend.read_status over a fake rollout (reuse the rollout-writing
      helper) returns the right turns/tokens/last_message and `None` for an
      unknown session; MockBackend stream+status; `get_backend` factory
      (known modes + ValueError on unknown). Each backend satisfies the protocol
      (an `isinstance`/structural check).
- [x] PROBE (best-effort, per the user): a small script/NOTES that attempts one
      long autonomous `codex exec` turn invoking `/flow` on a scratch project,
      recording unattended behaviour (approval mode, liveness, memory, failure).
      If codex is not authenticated in this environment, record that honestly as
      a pending live-env check rather than faking a result. Write findings to
      `tasks/20260720-221935/NOTES.md`.
- [x] Full check suite green; close-out.

## Definition of Done

- `AgentBackend` protocol with `CodexBackend` (exec + app_server) and
  `MockBackend` behind it; `get_backend` resolves by name
  (test: `get_backend_resolves_known_backends`).
- `CodexBackend.stream` runs in the agent's project cwd and resumes a given
  session (test: `codex_backend_stream_honors_cwd_and_session`).
- `CodexBackend.read_status` returns a normalized rollout snapshot (turns,
  tokens, last message) and `None` for an unknown session
  (test: `codex_backend_read_status_from_rollout`).
- The full suite passes (cmd: `nix develop --command bash -c "ruff check . &&
  mypy . && pytest -q"`).
- manual/probe: the unattended-`/flow` codex behaviour is recorded in NOTES.md
  (a live result, or an explicit "pending: codex not authed here").

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (Q2 rollout-tail; decisions 1,4; the
  "does a long codex-exec /flow turn behave unattended" open question).
- Depends on: 20260720-221929 (A1, landed 17bad00); built on the A0 supervisor.
- `read_status` keys off `session_id` alone (a rollout is uniquely named by it);
  `cwd` matters for LISTING a project's sessions, not for reading one known
  session. So `read_status(settings, session_id)` needs no cwd.
- SCOPE CUT for A2: this task builds the interface + backends + status; it does
  NOT rewire the existing `/api/chat/stream` orchestrator to route through
  `AgentBackend` (decision 4's "orchestrator is swappable"). The interface is
  designed so it CAN, but the rewire + the per-agent-run wiring
  (`CodexCliAgent` cwd, session binding) land in A3 where an agent actually runs,
  in one pass with the run mechanism.
- `state` in `BackendStatus` is optional and NOT set from the rollout (the
  rollout has no run-state); the live run-state comes from the A0 Supervisor and
  is merged in A3/A5. A2's status is the rollout-derived progress half.

## Close-out

What changed:
- New `scufris/backends.py`: `BackendStatus` model + `AgentBackend` protocol +
  `CodexBackend` (exec/app_server; `stream` delegates to agent.py runners with
  the A0 cwd seam, `read_status` reads the rollout via sessions.py) + `MockBackend`
  + `get_backend(name)` factory.
- `scufris/sessions.py`: new public `rollout_mtime(codex_home, session_id)` for
  the status `updated_at`.
- Tests `tests/test_backends.py` (6): stream cwd/session forwarding (exec +
  app_server mode selection), read_status from a rollout + None for unknown,
  MockBackend, factory (known + ValueError on "claude" until A2b), protocol
  conformance.
- Probe: `tasks/20260720-221935/NOTES.md` - live unattended codex turn confirmed
  working; corrected the "/flow" mis-generalization (codex is already agentic;
  /flow is Claude-Code-only) for A3.

Decisions / scope:
- Built the interface + backends + status ONLY; did NOT rewire the existing
  `/api/chat/stream` orchestrator through `AgentBackend` (deferred to A3 with the
  per-agent run wiring - it touches the StreamRunner fakes, one pass there).
- `read_status` keys off `session_id` alone (rollout is uniquely named by it),
  so no cwd needed - matches the codebase's existing read_context.
- `state` intentionally absent from the rollout-derived status; live run-state
  comes from the A0 Supervisor and is merged in A3/A5.
- Tested stream delegation by monkeypatching the runners (arg-forwarding +
  mode selection) rather than re-running a full fake-codex subprocess - the cwd
  subprocess wiring is already proven at the runner level in A0.

Difficulties: none significant. The probe's real value was the design
correction, not a red/green.

Result: 235 tests pass (+6), ruff + mypy clean; live probe green.

Self-reflection: the spike's "prompt invokes /flow" was a plausible-but-wrong
generalization that the probe caught early - a good example of the spike guidance
that a reasoned verdict about a dependency is a hypothesis until run live. A3's
plan must use a generic goal prompt, not a hard-coded /flow.
