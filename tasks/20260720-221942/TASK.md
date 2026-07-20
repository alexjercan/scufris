# A3: create-agent-with-goal end to end (background job, gated write, tracked state)

- STATUS: OPEN
- PRIORITY: 24
- TAGS: spike,agents

## Goal

First real vertical slice of the vision: create an agent bound to a project +
goal, launch it as a **background job** via the A0 supervisor (no held request,
no timeout), scoped to the project cwd via the A2 `AgentBackend` (the agent's
`backend` selects codex/claude), and track its lifecycle
(idle|running|blocked|done|error) by merging the A0 Supervisor run-state with the
A2 `read_status` rollout/session progress, surfaced by polling.

CORRECTION from the A2 probe (tasks/20260720-221935/NOTES.md): do NOT hard-code
"/flow" into the run. codex is already agentic (you hand it a goal prompt and it
runs its own loop); `/flow` is a Claude-Code-only skill. A3 hands each backend a
GENERIC GOAL PROMPT via `AgentBackend.stream(prompt=<goal>)`; each backend
realizes autonomy its own way. The `CodexCliAgent`-cwd wiring + the
`StreamRunner`-fake one-pass update (deferred from A0/A1) land here.

## Decisions (locked by the operator, 20260721)

- **Write scope: PLUMBING ON, DEFAULT OFF.** Build the `write_enabled` flag end
  to end - a write-enabled agent lifts the sandbox (codex: drop `--sandbox
  read-only` on the first turn per `codex-resume-rejects-sandbox`; claude:
  `--permission-mode acceptEdits`/equivalent) scoped to the project cwd - but v1
  agents default to READ-ONLY, and this flow does NOT exercise a live
  file-writing run. The write path is wired and unit-tested (the right flags are
  built for `write_enabled=True`), not live-verified. Flipping write on per agent
  is a deliberate later operator action.
- Pace: the operator PAUSED the flow after A2b (4/7). A3-A5 resume on their
  next go-ahead. This note is the on-disk pin so a fresh session resumes cleanly.

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (recommendation 3; decisions 2,3).
- Depends on: 20260720-221929 (A1, landed 17bad00), 20260720-221935 (A2, landed
  4d6850a); A2b (deb0ce9) gives the claude backend.
- Stepless direction-level task: run /plan before /work.
