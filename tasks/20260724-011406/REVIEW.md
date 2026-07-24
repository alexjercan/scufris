# Review: Make the app-server turn timeout an idle guard, not a wall-clock cap

- TASK: 20260724-011406
- BRANCH: bug/agent-idle-timeout

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

Out-of-context reviewer ran the suite (`python -m pytest tests/test_agent.py`
-> 30 passed) and verified: no orphaned `deadline`/`loop`; idle semantics on
every read path (setup handshake + streaming loop); the stall path still
raises and is caught -> `proc.kill()` + timeout StreamError; no new
unbounded-hang path. Confirmed both new tests are real guards - it read the
master streaming loop and reasoned that `slow_but_streaming` would fire
`remaining <= 0` mid-stream if reverted (in-session pass also observed this
test RED against the pre-fix code before the agent.py edit). Deferred siblings
(`mcp_server._CHAT_TIMEOUT`, opencode client) confirmed genuinely untouched.

No BLOCKER/MAJOR. Two non-blocking doc-accuracy findings, both addressed in
this round rather than deferred (cheap, and they concern code/text this task
touched or invalidated):

- [x] R1.1 (MINOR) scufris/config.py:162-169 - the rewritten `agent_timeout_seconds`
  docstring speaks purely in codex-app-server terms, but the same setting is
  also consumed by the opencode backend (`backends.py:717`) as an httpx client
  timeout with different semantics. Add a one-line note scoping the phrasing to
  the codex runner and acknowledging the opencode reuse.
  - Response: Fixed. Appended a sentence noting the opencode backend reuses this
    value as its own httpx client timeout (full alignment tracked by task-2,
    20260724-081804).
- [x] R1.2 (NIT) LESSONS.md:753 - the `sse-streaming-from-a-subprocess-in-fastapi`
  lesson still says to read stdout "with a wall-clock DEADLINE," the exact
  pattern this fix moved off. Update it so the ledger does not teach the bug.
  - Response: Fixed. Changed to "a per-read IDLE timeout (reset each line), not a
    per-turn wall-clock deadline" with a pointer to task 20260724-011406.

No open `manual:` items block APPROVE; the task's `manual:` DoD (read the
docstring diff) batches to the flow Finish checkpoint.
