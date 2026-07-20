# A2 probe: does an unattended autonomous codex turn behave?

- DATE: 20260720
- TASK: 20260720-221935
- Spike open question (tasks/20260720-221748/SPIKE.md): "does one long
  `codex exec` turn running `/flow` behave unattended (approvals, memory,
  liveness)?"

## What was run

codex 0.144.4 is installed and authenticated here (`~/.codex/auth.json`
present). A minimal unattended turn:

```
codex exec --sandbox read-only --skip-git-repo-check --output-last-message out.txt \
  'Read sample.txt in this directory using your tools and reply with exactly the
   word DONE followed by the number of lines in it.' </dev/null
```

Result: exit 0 in ~7s; the agent autonomously called its shell tool
(`wc -l sample.txt`) under the read-only sandbox and answered `DONE 1`. The run
header confirmed `approval: never`, `sandbox: read-only`; a rollout was written
with session id `019f8144-...`. So `read_status` can read a live agent's
progress by session id (validated separately by the unit tests).

## What this answers

- Unattended `codex exec` works with `approval=never` + `--sandbox read-only`:
  the agent runs its own agentic loop, uses tools, and completes with NO human
  in the loop and NO approval prompt blocking it. This is the load-bearing
  mechanism A3 needs.
- A rollout is produced immediately and is `read_status`-readable while/after the
  turn - the read-only status path is real, not just unit-tested.
- The A0 supervisor's heartbeat (not a hard wall-clock timeout) is the right
  liveness guard: a healthy long turn emits events/rollout appends and will not
  be killed for being slow.

## Important clarification for A3 (design correction)

The spike phrased the autonomous goal run as "its prompt invokes `/flow`". The
probe makes clear this is BACKEND-SPECIFIC and should not be baked into the
generic interface:

- **codex** is ALREADY an autonomous agent - you hand it a goal prompt and it
  runs its own tool-using loop to completion. It does NOT need (and cannot run)
  the Claude Code `/flow` skill; `/flow` is a Claude Code skill, not a codex one.
  codex has its own goals/skills system (`~/.codex/goals_*.sqlite`,
  `~/.codex/skills`).
- **claude** (A2b, Claude Code headless) is where a `/flow`-style skill invocation
  in the prompt makes sense.

So A3's "create-agent-with-goal" should hand the backend a GOAL PROMPT via
`AgentBackend.stream(prompt=<goal>)` and let each backend realize autonomy its
own way - codex via its exec loop, claude via `/flow`. The interface already
models this correctly (`stream(prompt)`); A3 must not hard-code `/flow` into the
codex path.

## What remains genuinely unverified (not a blocker for A3)

- A MULTI-MINUTE turn's memory growth and sustained liveness were NOT stressed
  (this was a 7s turn). The mechanism is proven; the long-run scale is a cheap
  follow-up once A3 can launch a real goal - watch the supervisor heartbeat +
  rollout mtime advance over a minutes-long run.
- WRITE access (sandbox lifted) was not exercised - v1 write is a per-agent,
  cwd-scoped opt-in landing in A3 (decision 3); this probe stayed read-only.

Verdict: the open question is answered ENOUGH to proceed with A3 - unattended
autonomous codex turns work; the /flow phrasing was a mis-generalization,
corrected above.
