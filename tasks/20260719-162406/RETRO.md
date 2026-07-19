# Retro: Agent chat panel in the dashboard

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- Probing `codex exec --json` FIRST (one tiny call) revealed the event shape
  (`thread.started` with a `thread_id`, turn-level not token-level), which set an
  honest scope up front: turn-based replies + resume-based continuity, no faked
  streaming. No rework from a wrong assumption.
- The live two-turn check (codeword BANANA -> recalled, then reset -> forgotten)
  is exactly the kind of end-to-end proof a chat feature needs; it also caught
  the resume/sandbox bug that all the unit tests missed.
- Reusing the unchanged `Agent` interface meant the backend/UI only grew a chat
  endpoint and a panel - the runtime work from the prior task carried straight
  through.
- Chat text uses `textContent` (not `innerHTML`), so the XSS surface flagged for
  the stat cards was avoided here by construction.

## What went wrong / friction

- First live turn-2 failed: `codex exec resume` inherits the session sandbox and
  REJECTS a repeated `--sandbox`, unlike `codex exec`. Only the live run exposed
  it (the fake-codex test ignored unknown flags). Fixed by passing `--sandbox`
  only on turn 1. Lesson: subcommand flag sets differ; the fake can't catch a
  real CLI's arg validation.
- Kept discovering codex CLI specifics by running it - worth it, but each probe
  is a metered call.

## Lessons

- `codex-resume-rejects-sandbox`: `codex exec resume` inherits the original
  session's sandbox and errors on a repeated `--sandbox`; pass sandbox (and other
  session-scoped flags) only on the FIRST turn, not on resume. Only a live run
  catches this - a fake that ignores unknown args won't.
- `probe-cli-json-shape-before-scoping-streaming`: check a CLI's `--json` event
  granularity before promising "streaming"; `codex exec` is turn-level, so chat
  is honestly turn-based (pending -> full reply), not token-streamed.

## Follow-ups

- Visible tool activity in the chat (show when the agent runs a tool) lands with
  the MCP tool server (tatr 20260719-162419).
- Per-client conversations (today one global conversation is shared across tabs)
  only if multi-client is ever wanted - fine for single-user now.
