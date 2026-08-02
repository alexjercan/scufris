# Extract the orchestrator-turn and agent-run services

- PRIORITY: 71
- TAGS: refactor, v0.2.0, agents, backend, telegram
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -
- PARENT: 20260729-102145
- DEPENDS ON: 20260801-100425

## Story

As a maintainer, I want one transport-independent orchestrator-turn service and
one agent-run service, so that the landing chat, Telegram, and the wake bridge
stop each implementing their own version of starting and finishing a turn.

## Steps

- [ ] Add failing tests asserting that the landing chat, Telegram, and the wake
      bridge all launch through one turn service with typed inputs and results.
- [ ] Extract the orchestrator-turn service from `/api/chat`,
      `/api/chat/stream`, `/api/chat/reset`, and the Telegram turn path in
      `scufris/telegram/turn.py`. No FastAPI or Telegram rendering concerns
      inside it.
- [ ] Extract the agent-run service owning launch, resume, cancel, status,
      completion, outcomes, and supervisor interaction, from the
      `/api/agents/{id}/{run,cancel,status,chat,fork,acknowledge,report_back,request_input}`
      handlers and `build_telegram_callbacks`.
- [ ] Keep the streaming and SSE behavior identical: the service returns
      events, the transport renders them.
- [ ] Preserve shutdown, background task, and callback behavior; the run
      service owns the supervisor handle rather than a closure in `create_app`.
- [ ] Leave the routes in place, delegating to the services; the router split
      is the successor task.

## Definition of Done

- All three transports launch through the same service
  (test: `test_orchestrator_transports_share_turn_service`).
- Run lifecycle is owned by one service for every caller
  (test: `test_agent_run_lifecycle_is_owned_by_the_run_service`).
- The services carry no transport imports
  (cmd: `! rg -n "fastapi|telegram" scufris/services/`, path adjusted to the
  chosen module location).
- Streaming and SSE behavior is unchanged
  (test: `test_chat_stream_events_are_unchanged`).
- Existing API and browser suites pass without drift
  (cmd: `python -m pytest && cd web && npm run ci && npm run test:e2e`).

## Notes

- Epic: 20260729-102145.
- Depends on the route characterization task; its route-contract test is the
  gate for this one too.
- Do not invent the future conversation schema here. The required seam is one
  transport-independent orchestrator service that 20260729-220835 can place a
  durable conversation around later.
- Move models and helpers only when their ownership becomes clearer. A
  mechanical one-file-to-many split with the same coupling is not the goal.
