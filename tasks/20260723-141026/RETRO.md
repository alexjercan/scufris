# Retro: tool console reaches its own server; revert pending path

- TASK: 20260723-141026
- BRANCH: fix/tool-console-loopback
- REVIEW ROUNDS: 1 (APPROVE, clean)

## What went well

- Peeled the onion instead of stopping at the first plausible cause. The report
  looked like a route bug, then a one-line base default - but each fix exposed the
  next layer, and the REAL blocker (a loop-blocking self-loopback) was only visible
  after the first two. Booting the actual scenario (dashboard on a non-default
  port, tool console POST) at the end confirmed all three layers at once.
- The real-socket integration test earned its keep and was sabotage-verified both
  by me and the reviewer: revert the off-loop fix -> `httpx.ReadTimeout`. This is a
  bug a respx/ASGITransport test would have PASSED while production hung.

## What went wrong

- The `_ensure_api_base` unit test LEAKED `SCUFRIS_API_BASE` into `os.environ`,
  reddening 19 unrelated `mcp_server` respx tests. Root cause: `_ensure_api_base`
  mutates `os.environ` directly (via `setdefault`), which monkeypatch does NOT
  track - so monkeypatch's teardown restore of a LATER `setenv` reverted to the
  leaked value, not to absent. Fixed with an explicit snapshot/restore in a
  `finally`. Lesson: a test of a function that raw-mutates `os.environ` cannot lean
  on monkeypatch for cleanup.
- The route move I did last task (20260723-120507) turned out to be aimed at a
  red herring - the operator's 404 was the base bug, not route shadowing. I
  correctly proved the ordering was fine THEN, but did not chase the "why is the
  tool hitting a build without the route" far enough to find the base default. The
  path move was still a defensible robustness step, but the real fix was elsewhere;
  I reverted it here per the operator's call.

## What to improve next time

- When a symptom is "the client reaches a server that lacks the route", check the
  client's TARGET (base URL / port) before assuming a routing bug - especially for
  in-process HTTP-loopback tools whose base has a default.
- For any in-process tool/handler that makes a BLOCKING call which could loop back
  to the same server, run it off the event loop and prove it with a REAL socket -
  respx/ASGITransport return instantly and hide the deadlock.

## Action items

- [x] Ledger: `os-environ-setdefault-in-test-leaks-past-monkeypatch` (x1) and
  `self-loopback-blocking-call-needs-off-loop-run-and-a-real-socket-test` (x1).
- Returns to BC4 (the wake bridge) next.
</content>
