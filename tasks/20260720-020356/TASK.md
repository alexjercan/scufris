# Stream assistant tokens end-to-end in the browser

- STATUS: CLOSED
- PRIORITY: 55
- TAGS: bug, agent, ui, streaming
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Symptom

With the app_server backend, the reply appeared as one final message in the UI
instead of streaming token-by-token, even though the API was clearly producing
deltas.

## Diagnosis (layer by layer)

The buffering could have been at any layer; each was measured, not guessed:

1. `codex app-server` streams - 252 deltas over 5.5s (timestamped probe).
2. scufris backend + StreamingResponse + middleware + uvicorn stream over a real
   TCP socket - 179-186 deltas over ~3s. (In-process httpx ASGITransport buffers,
   which produced two false "buffered" readings until switching to a real socket.)
3. The frontend renders incrementally (jsdom tests feeding chunk-by-chunk).
4. Root cause 1: **webpack-dev-server** (`:8090`, `npm run serve`) injects the
   gzip `compression` middleware by default (`compress: true`) in front of the
   proxy; it buffers sub-1KB SSE chunks to the end. Proven with an A/B: with
   `Accept-Encoding: gzip`, `compress:true` -> all chunks at 1.52s;
   `compress:false` -> 0.3s apart. `:8000` (no dev server) was fine all along.
5. Root cause 2 (latent): the render used a single queued `requestAnimationFrame`
   that `onDone`'s `renderLog` could clobber (detaching the bubble) before it
   painted, so a buffered burst showed nothing until the end.

## Fix

- `web/webpack.config.js`: `compress: false` on devServer - the actual fix for
  `:8090`.
- `web/src/agent-view.ts`: eager, time-throttled (~50ms) synchronous render; the
  first token paints immediately and the last frame can't be lost to the onDone
  race (dropped rAF).
- `scufris/app.py`: SSE anti-buffering headers (`Cache-Control: no-cache`,
  `X-Accel-Buffering: no`, `X-Content-Type-Options: nosniff`) + a leading
  comment/padding flush frame; and `_NoCacheStaticFiles` so a rebuilt,
  non-hashed bundle is never served stale.

## Tests

- `test_app.py`: SSE response carries the anti-buffering headers + leading `:`
  frame; static bundle served with `Cache-Control: no-cache`.
- `agent-view.test.ts`: parseSseFrames ignores the padding frame; the pending
  bubble renders token deltas incrementally (eager first paint + throttled flush).

## Definition of Done

- [x] `:8090` (dev server) streams token-by-token like `:8000`.
- [x] Render can't be clobbered by the done race; first token is immediate.
- [x] Backend ships anti-buffering headers; static bundle revalidates.
- [x] 122 pytest + 58 frontend green.
