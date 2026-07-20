# A0: de-singleton the agent runtime (per-agent cwd + lock)

- STATUS: OPEN
- PRIORITY: 30
- TAGS: spike,agents,refactor

## Goal

Foundation / gating refactor for the multi-agent orchestrator. Generalize the
singleton assumptions in the agent runtime so more than one agent, on more than
one project cwd, can coexist:

- Session listing hard-filters `cwd == os.getcwd()` (sessions.py:255-259) - make
  it per-agent cwd instead of the server's single cwd.
- The global `chat_lock = asyncio.Lock()` (app.py:303) serializes every turn -
  make locking per-agent so two agents can run concurrently.
- Turns inherit `os.getcwd()`; pass the agent's project cwd to the codex
  subprocess (`-C`/cwd) instead.

This is the refactor that gates A1-A5. No new user-facing feature by itself.

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (option C, A runner; "the singleton /
  one-cwd assumption" blocker).
- Stepless direction-level task: run /plan to break into steps before /work.
