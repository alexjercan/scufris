# A0: agent runtime foundation (de-singleton + background supervisor, no request timeout)

- STATUS: OPEN
- PRIORITY: 30
- TAGS: spike,agents,refactor

## Goal

Foundation / gating refactor for the multi-agent orchestrator. Two things
together (spike revision 1, decisions 1-2):

- **De-singleton.** Session listing hard-filters `cwd == os.getcwd()`
  (sessions.py:255-259); make it per-agent cwd. Turns inherit `os.getcwd()`;
  pass the agent's project cwd to the subprocess (`-C`/cwd) instead.
- **Background execution + event bus (ADR-001).** The global
  `chat_lock = asyncio.Lock()` (app.py:303) runs every turn inside the held HTTP
  request. Replace it with an in-process supervisor that runs agent subprocesses
  as background jobs, with a concurrency cap acting as the queue (agents past the
  cap wait). Split the run from its stream: `POST .../run` enqueues and returns a
  run id immediately; each worker publishes normalized `StreamEvent`s to a
  per-agent event bus (fan-out ring buffer, backed by the durable rollout/session
  log). `GET .../events` is a thin SSE subscriber that replays the buffer and
  streams live (drop-safe, `Last-Event-ID`); `GET /api/agents` polls coarse
  status. Replace the 120s `agent_timeout_seconds` hard kill with a per-agent
  budget + liveness/heartbeat that only catches a genuinely stuck subprocess.

SSE is kept (delivery), workers give concurrency + no timeout (execution) - the
two are orthogonal (ADR-001). Gates A1-A5. No new user-facing feature alone.

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (decisions 1-2; "singleton / one-cwd"
  blocker; execution model).
- No external broker (no Redis/Celery) - lightweight in-process supervisor.
- Stepless direction-level task: run /plan before /work.
