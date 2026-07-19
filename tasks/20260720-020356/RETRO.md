# Retro: stream tokens end-to-end in the browser

- DATE: 20260720
- VERDICT: fixed (multi-layer), verified with real socket + A/B probes

## What went well

- Measuring each layer instead of theorizing found the real culprit. The bug
  "felt" like a backend/streaming-code problem; it was the dev-server's gzip
  middleware. Timestamped probes (codex, real socket, compression A/B) turned a
  guessing game into a decision tree.
- The `curl` vs browser split was the key tell: curl (local, no dev proxy, no
  Accept-Encoding) streamed while the browser did not - which pointed straight at
  something in the browser's path that curl skips.

## What went wrong / friction

- Burned two iterations on `httpx.ASGITransport`, which buffers in-process, so my
  first "the stack buffers" and "the middleware buffers" readings were false. A
  real TCP socket is the only trustworthy way to test streaming.
- The `compression` middleware only buffers when it actually compresses (client
  sends `Accept-Encoding: gzip` AND the body is below its 1KB threshold, so it
  holds chunks waiting to decide). My first A/B used a Node client with no
  Accept-Encoding and wrongly showed "streams" - had to add the header to
  reproduce.

## Lessons

- `test-streaming-over-a-real-socket-not-asgitransport` - httpx ASGITransport
  (and TestClient) buffer the response body, so they can NEVER prove SSE streams
  in real time. Spin a real uvicorn on a port and read with a socket client.
- `webpack-dev-server-compression-buffers-sse` - dev server defaults
  `compress: true`, injecting gzip `compression` in front of the proxy; it
  buffers sub-1KB SSE to the end. Set `compress: false` for any SSE endpoint.
- `dont-gate-streaming-render-on-a-single-raf` - a lone queued rAF can be
  clobbered by a later synchronous re-render (onDone -> renderLog detaches the
  node). Paint eagerly + time-throttle instead, so a buffered burst still shows.
- `curl-streams-browser-doesnt-suspect-the-path-between` - when curl streams but
  the browser buffers, the difference is the transport in between (a proxy /
  dev-server / compression), not the server or the app code.

## Follow-ups

- None blocking. If a reverse proxy is ever put in front in prod, it needs its
  own no-buffering config (nginx `proxy_buffering off`, etc.); the
  `X-Accel-Buffering: no` header already covers nginx.
