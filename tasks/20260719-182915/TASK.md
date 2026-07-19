# Sparkline history: background sampler + card mini-graphs (deferred)

- STATUS: OPEN
- PRIORITY: 5
- TAGS: feature,backlog,dashboard

## Goal

(Deferred follow-up.) Add live time-history to the stats cards: a background
sampler that polls the collector on a timer into a ring buffer, exposed as recent
history so the cards can render btop-style sparklines (cpu / gpu / mem / net over
the last few minutes).

## Notes

- Spike: tasks/20260719-180507/SPIKE.md. User chose "current-first, history
  later" - this is the explicit follow-up.
- The collector is designed as the seam: keep all sampling inside it so a
  `BackgroundSampler` can call `sample()` on a timer into a bounded ring buffer
  (per-metric series) without changing the per-request path.
- Backend: an async background task (FastAPI lifespan) samples every N seconds;
  expose history via an endpoint (e.g. `GET /api/history`) or embed recent series
  in the payloads. Bound memory (ring buffer length).
- Frontend: sparkline/mini-graph widgets on the existing cards (cpu, gpu, mem,
  net), themed. Keep the current live numbers; graphs are additive.
- Depends on the richer host metrics (tatr 20260719-182846) so there is more to
  graph. Do AFTER the current-values work lands.
