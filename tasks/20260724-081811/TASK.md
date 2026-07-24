# Spike: auto-retry an agent turn on genuine stall / transient app-server failure

- STATUS: OPEN
- PRIORITY: 40
- TAGS: spike,agent,codex

## Story

As an agent operator, I want a turn that hits a GENUINE stall or a transient
`codex app-server` failure (closed pipe, setup error, idle-timeout) to retry
automatically instead of surfacing a hard error, so that a one-off hiccup does
not abort a run.

Deferred out of the idle-guard goal (umbrella 20260724-081616) by explicit user
decision: with the idle guard in place, a progressing turn never times out, so
the reported bug is fixed WITHOUT retry. Retry is a separate, larger design
concern captured here so it is not silently dropped.

## Open questions (spike this first)

- Retry only on SETUP-phase failure (before any token streamed) vs mid-stream?
  Mid-stream retry needs `thread/resume` continuation and risks
  duplicated/partial output.
- Bounded retries + backoff policy; which failures are retryable
  (idle-timeout vs `app-server closed` vs thread-setup error).
- Where retry lives: the runner (`_stream_app_server`), the supervisor, or the
  backend wrapper.
- UI/backend behaviour: swallow the transient error and continue vs show a
  "retrying" indicator.

## Steps

- [ ] `/spike` the retry design (setup-only vs mid-stream, policy, placement).
- [ ] Plan into concrete tasks once the direction is chosen.

## Definition of Done

- A durable SPIKE.md recording the retry direction + seeded implementation
  tasks. (manual: user reviews the spike)

## Notes

- Depends on: 20260724-011406 (the idle-guard fix) landing first.
- NOT part of the idle-guard umbrella; own follow-up.
