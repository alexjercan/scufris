# Decision: An `orchestrator/` package with typed refusals and completion hooks

- DATE: 20260803-070000
- STATUS: ACCEPTED
- TASK: 20260801-100441
- TAGS: refactor,backend,agents,telegram,testability

## Context

`_launch_agent_turn` (`app.py:1557-1735`) is already the single seam every turn
goes through: `/api/agents/{id}/run|chat|fork`, `/api/chat`, `/api/chat/stream`,
`build_telegram_callbacks`, the wake bridge's `_wake_launch`, and the
host-decision delivery `_deliver_decision`. It is not duplicated - it is
trapped. It is a `create_app` closure over `supervisor`, `agent_runs`,
`launching_runs`, `agents`, `projects`, `settings`, `reasoning_store` and
`wake_bridge`, so exercising a turn costs a state directory, a SQLite database,
two stores, a supervisor and a lifespan.

Extracting it has to answer four things: where the service lives, how it refuses
without FastAPI, what happens to the image attachment that only one transport
sends, and how the wake bridge - which the service must call and which calls the
service - is wired without a construction cycle.

## Decision

**1. The services live in `scufris/orchestrator/`, not `scufris/services/`.**

Three modules: `errors.py`, `runs.py` (`AgentRunService`), `turn.py`
(`OrchestratorTurnService`). Every service in this tree is named for what it
serves, never for the fact that it is a service - `host_approvals.py`,
`agent_diagnostics.py`, `hostconfig/service.py`, `wake.py`. `orchestrator/` also
matches the vocabulary `scufris/README.md` section 4 already uses, and it is the
seam 20260729-220835 will put a durable conversation around. A package rather
than two top-level modules because both share `errors.py` and the turn service
holds the run service.

**2. Transport refusals become typed errors, translated in `api/errors.py`.**

`RunAlreadyActive` (409), `NoActiveRun` (404), `AgentDisabled` (503),
`AgentProjectMissing` (422), `TurnFailed` (503), `TurnEndedWithoutReply` (500)
live in `orchestrator/errors.py`; the status mapping goes beside
`hostd_http_error` in `api/errors.py`, the module that already exists to hold
exactly this translation for the host surface.

This is a deliberate behavior change at three call sites. `_deliver_decision`,
`_wake_launch` and the Telegram `on_message` all `except HTTPException` to mean
"busy", so a 404 (agent deleted mid-launch) or a 503 (agent disabled) is
currently swallowed and reported as an already-active race. `except
RunAlreadyActive` is strictly narrower and lets the rest propagate.

**3. Image decoding stays at the HTTP transport.**

`_write_image_to_temp` and the single-error-frame SSE response stay in the
`/api/chat/stream` route. The service keeps taking `image_paths` and the
`on_done` cleanup callback it takes today, so the tempdir's ownership - the
turn, not the relay - is unchanged.

**4. Run completion fans out through registered hooks.**

`AgentRunService.on_complete(hook)`, in the shape `HostApprovalService.on_proposed`
already establishes. `create_app` builds the service, then the `WakeBridge` over
`runs.launch` / `runs.active`, then registers `wake_bridge.on_run_complete` and
`_drain_deferred_decision` in that order. Hooks run in registration order at the
end of `persist`, after `mark_finished` and past the serialize-key release.

## Alternatives considered

- **`scufris/services/`** (the name the Steps sketched). Grouping by layer says
  the same thing about every module inside it, and gives the next service no
  rule about whether it belongs there or at the top level. Rejected for a
  domain name.
- **Two top-level modules, `agent_runs.py` + `orchestrator_turn.py`.** Rejected:
  they share the error vocabulary, and `errors.py` would have to be a third
  top-level module in an already crowded namespace.
- **Keep raising `HTTPException` from the service.** Rejected: it is the one
  thing the DoD forbids, and it is what makes the turn path untestable without
  `create_app`.
- **Return a result union instead of raising.** Rejected: every caller is an
  `await` inside a route or a completion callback that wants the failure to
  propagate; a union adds an `if result.failed` at six call sites to reproduce
  what `raise` already does.
- **Move image decoding into `OrchestratorTurnService.stream`.** Rejected: it
  puts a base64/MIME concern behind the same door as the Telegram and
  wake-bridge callers, neither of which sends images, and its failure mode is an
  SSE frame the service is not allowed to know how to build.
- **Constructor-injected completion callbacks.** Rejected: the wake bridge needs
  the run service to exist first, so one of the two would be assigned after
  construction regardless. A registration method is honest about that instead of
  leaving a settable attribute.
- **Move the routes onto routers in this task.** Rejected: the task explicitly
  defers the router split to 20260729-103712, and doing both at once would put
  the route-contract characterization test under two independent moves at the
  same time.

## Consequences

- A turn can be driven with no FastAPI app, no database and no lifespan; the
  four new tests do exactly that.
- Three callers stop treating every launch failure as "busy". A wake or host
  decision aimed at a deleted or disabled agent now surfaces instead of being
  reported as a race.
- `build_telegram_callbacks` leaves `scufris/app.py` for
  `scufris/telegram/orchestrator.py`. `tests/test_telegram_app.py`'s import
  moves with it; there is no compatibility re-export.
- `app.state.runs` is a new published key. `app.state.supervisor` stays - the
  route-contract test pins the exact key set and `tests/test_app.py` calls
  `list_runs()` on it - so the route-contract expectation gains one key.
- `app.py` drops from 2923 to roughly 2500 lines. It stays on the file-size
  allowlist; 20260729-103712 removes the entry.
- `orchestrator/runs.py` lands close to the 600-line `SOURCE_CAP`. If the
  successor task grows it, the split is already drawn along
  launch / lifecycle / outcomes.
