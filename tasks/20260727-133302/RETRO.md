# Retro - 20260727-133302 (>64 KiB app-server line errors the run)

## What was delivered

A shared `STREAM_READ_LIMIT` (8 MiB) in `scufris/agent.py`, passed as `limit=`
to `create_subprocess_exec` at both the codex `app-server` launch
(`_stream_app_server`) and `ClaudeBackend.stream`, so a single backend line over
asyncio's 64 KiB default no longer raises `ValueError`. All three `readline()`
sites are wrapped so an over-limit line becomes a clean `StreamError` (streaming
loops) / `AgentUnavailable` (handshake) instead of an uncaught exception.
Follow-up `20260727-140443` filed for surfacing `run.error` detail through
`agent_status`.

## What went well

- The task doc had already nailed the root cause (default 65536 readline limit,
  three exact line/col sites), so the work was aimed, not exploratory. Verifying
  the code matched the doc before planning took minutes.
- Reproduce-first paid off: the real fake-app-server test emitting one 200 KB
  line reproduced the exact `ValueError` out of the async generator BEFORE any
  fix, then went green with `limit=` - a true red->green pin, not a
  written-after-the-code test.
- Driving the defense-in-depth test by monkeypatching `STREAM_READ_LIMIT` tiny
  and launching a REAL subprocess (not a mock) meant the wrapping branch is
  exercised by real asyncio `readline` raising the real `ValueError` - high
  fidelity, and it confirmed both the limit and the wrapping are load-bearing.
- The design matched existing precedent: yielding `StreamError` and ending in
  `DONE` is exactly what the idle-timeout path already does, so the change added
  no new run-state semantics. Confirming that (reading `_drain`) up front avoided
  a spurious "does this regress the ERROR state?" worry.

## What went wrong / friction

- First `nix flake check` failed on an unused import: I imported
  `STREAM_READ_LIMIT` into `tests/test_agent.py` but referenced it only via the
  monkeypatch STRING `"scufris.agent.STREAM_READ_LIMIT"`, so ruff F401 flagged
  it. Lesson reinforced: an import used only inside a monkeypatch path string is
  NOT a real reference - either use the symbol or don't import it.
- `ruff format --check` reported drift in files I never touched
  (`agent_store.py`, `test_telegram.py`) - pre-existing formatter-version drift.
  The `format-only-the-files-you-edited-not-whole-dirs` ledger lesson (x3) was
  exactly right: I scoped `ruff format` + `ruff check --fix` to my 4 files only,
  keeping the diff clean.

## What to do differently next time

- When adding a module constant that a test will only reach via a monkeypatch
  string, skip the import entirely (or assert against it directly) - don't add a
  dead import that the gate rejects.
- Keep running the scoped WRITING formatter (`ruff format <touched files> &&
  ruff check --fix <touched files>`) as a routine pre-gate step, since the flake
  gate is lint-only and will not reformat but WILL reject an I001 the format pass
  leaves behind.

## Lessons candidates (for /lessons at Finish)

- `subprocess-line-reader-needs-explicit-limit` (NEW): any
  `asyncio.create_subprocess_exec` whose stdout is consumed with `readline()`
  needs an explicit `limit=` - the 64 KiB default raises a bare `ValueError` on
  a longer line, which for an LLM app-server/stream-json frame (big `rg`, `tatr
  ls`, file dump) is a routine, not exceptional, occurrence. Both scufris
  backends had this latent. Wrap the readline too so the overflow is a
  diagnosable event, not an opaque crash.
- `import-used-only-in-monkeypatch-string-is-unused` (NEW, minor): a symbol
  referenced solely through a `monkeypatch.setattr("mod.NAME", ...)` STRING is
  not an import use; ruff F401 rejects the import. Reinforces the existing
  format/lint-gate lessons.
