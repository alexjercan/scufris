# Review - 20260727-133302 (>64 KiB app-server line)

## Round 1 - out-of-context reviewer

- VERDICT: APPROVE (1 round)

Reviewed `git diff master...HEAD` on `fix/appserver-stream-limit` plus the
surrounding code in `scufris/agent.py` (`_stream_app_server`, `_appserver_call`)
and `scufris/backends.py` (`ClaudeBackend.stream`).

### Correctness confirmed

1. `limit=` semantics are load-bearing and correct: `create_subprocess_exec(...,
   limit=N)` forwards into the stdout `StreamReader` buffer ceiling, so an 8 MiB
   limit genuinely prevents the `ValueError` for lines up to that size. The
   `except ValueError` wrapping is secondary hardening; both are present.
2. `except ValueError` is narrowly scoped - each `try` wraps ONLY the
   `await ...readline()` (and its `wait_for`). `json.loads` /
   `_parse_event_line` / `_appserver_event` are outside the try, so a JSON
   `ValueError` cannot be swallowed. `wait_for` raises `TimeoutError`, not
   `ValueError`, so the idle-timeout path is untouched.
3. Handshake -> StreamError chain intact: `_appserver_call` re-raises as
   `AgentUnavailable`, which `_stream_app_server` maps to a yielded `StreamError`.
4. Proc cleanup correct: over-limit branches `proc.kill()` then `return`; the
   `finally` guards on `returncode is None` and reaps with `await proc.wait()`.
   No leak, no double-reap hang.
5. Tests are genuine red/green: the survive-test uses a REAL fake subprocess
   emitting a 200 KB line (pre-fix raises `ValueError` out of the generator);
   the over-limit test monkeypatches `STREAM_READ_LIMIT` BEFORE launch so a real
   asyncio `readline` raises the real `ValueError`. High fidelity.
6. Run-state semantics match the existing idle-timeout precedent (yield
   `StreamError`, end in `DONE`).

### Completeness

The two other `create_subprocess_exec` sites (`agent.py` `login()`) do not read
stdout line-by-line, so they correctly need no `limit=`. All three `readline`
sites are covered. Scope complete.

### NITs

- (pre-existing, out of scope) The claude streaming loop lacks the codex loop's
  `wait_for(..., timeout=idle)`, so a wedged claude process can hang on
  `readline()`. Unchanged by this diff; noted for a future task.
- (fixed in round 1) test helper param `biline_bytes` -> `bigline_bytes` to
  match the `BIGLINE` constant.
- The orchestrator-visibility gap (a backend `StreamError` reaches
  `agent_status` without its `detail`) is correctly split into follow-up
  `20260727-140443`.

No BLOCKER / MAJOR / MINOR issues. Approved.
