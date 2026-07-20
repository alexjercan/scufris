# Retro: A2b ClaudeBackend behind the AgentBackend interface

- TASK: 20260720-223938
- BRANCH: feature/claude-backend (landed deb0ce9)
- REVIEW ROUNDS: 1 out-of-context APPROVE (1 MINOR stderr-deadlock fixed, 2 NIT) + in-session round 2

## What went well

- Probing the real `claude -p --output-format stream-json` output AND the session
  transcript BEFORE writing any parser (the lesson promoted in A2) was the whole
  game: the parser was written against captured reality, and the
  session-findable-by-id-glob insight - which kept the `AgentBackend` interface
  UNCHANGED - came from inspecting the actual files, not guessing.
- A2b did its job: a genuinely different backend (different CLI, whole-message
  output vs codex's rollout, a different on-disk store) slotted behind the
  identical protocol with zero interface changes. That is the real proof the A2
  interface was not accidentally codex-shaped.
- Testing the parser against captured real lines + the subprocess via a
  monkeypatched proc kept the tests fast and CI-independent of a live claude.

## What went wrong

- Left `stderr=PIPE` undrained on the claude subprocess - a latent deadlock if
  claude ever wrote >64KB to stderr while we read stdout. Root cause: copied the
  codex streamers' PIPE-both-streams shape without noting that codex DRAINS
  stderr (via communicate/readline) whereas the claude loop only reads stdout.
  The reviewer caught it; fixed to DEVNULL.

## What to improve next time

- When spawning a subprocess and reading only ONE stream, set the other to
  DEVNULL (or drain it) deliberately - an undrained PIPE is a deadlock waiting
  for a chatty process. Don't inherit a "PIPE both" shape from code that drains
  both.

## Action items

- [x] stderr deadlock fixed (DEVNULL); error detail now includes the subtype.
- No new ledger entry: the stderr-pipe point is a well-known subprocess gotcha,
  and the probe-first lesson is already promoted; keeping the ledger terse.
