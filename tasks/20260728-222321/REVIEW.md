# Review: Telegram read-only /settings subcommands + /stats

- TASK: 20260728-222321
- BRANCH: feature/telegram-settings-cmds

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Check suite (run in the worktree by the reviewer and re-confirmed in-session):
`python -m pytest` 582 passed; `ruff check .` clean; `mypy .` clean;
`nix flake check` green (ruff + mypy + pytest). DoD proofs 1-4 and 6 pass;
DoD 5 (read-only) verified by inspection - no writes/toggles/PATCH/exec added.

Independently re-derived load-bearing claims: (a) the sync-provider blocking
below - `collector.sample()`/`read_usage()` are synchronous and awaited inline
in `_dispatch`, so they block the event loop, matching the task's own "can a
slow provider block the poll loop" concern; (b) the DTO relocation is
behavior-preserving - the three models moved byte-identically to
`scufris/mcp_models.py`, are re-imported into `app.py`'s namespace and still used
as the same endpoint response models, and the full `test_app.py` endpoint suite
passes, so the OpenAPI shape is unchanged.

- [x] R1.1 (MINOR) scufris/app.py - the `usage()` and `stats()` providers call
  the SYNCHRONOUS `read_usage(...)` (an `rglob` + `stat` + JSONL parse over every
  rollout) and `collector.sample()` (psutil) directly with `await`, and
  `_handle_settings`/`/stats` run INLINE in `poll_once` (unlike turns, dispatched
  via `create_task`). So a `/settings usage` or `/stats` on a box with many
  rollouts blocks the whole event loop (stalling the next `getUpdates` and any
  concurrent turn's streaming edits) until it returns. Wrap the sync readers in
  `asyncio.to_thread(...)` so they no longer block the loop.
  - Response: fixed in round 2 - both providers now `await asyncio.to_thread(...)`.

- [ ] R1.2 (NIT) scufris/telegram.py `_send_settings` - on a MarkdownV2 rejection
  the plain-text fallback re-sends the raw GFM `body` (literal `**Title**` +
  triple-backtick fences), so a degraded reply shows raw markdown syntax. This
  correctly preserves the never-drop-a-reply guarantee and matches the turn-reply
  pattern; left as-is (stripping the markers for the plain fallback is optional).
  - Response: acknowledged; left as-is by design (parity with the turn reply's
    fallback, and the fence markers are harmless plain text).

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (trivial diff - a two-line `asyncio.to_thread` wrap
  addressing R1.1; no behavior change beyond off-loading two sync reads)

R1.1 addressed: `_build_telegram_settings_ops`'s `usage()` and `stats()`
providers now `await asyncio.to_thread(read_usage, ...)` / `to_thread(
collector.sample)`, so the synchronous rollout scan and psutil read run off the
event loop and can no longer stall the poll loop. R1.2 (NIT) left as-is by
design. Re-verified: `python -m pytest` 582 passed, `ruff check .` clean,
`mypy .` clean.
