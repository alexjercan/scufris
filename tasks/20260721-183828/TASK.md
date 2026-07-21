# Bug: codex agent in auto/edit permission mode still runs read-only (sandbox not applied)

- STATUS: OPEN
- PRIORITY: 38
- TAGS: bug,agents,backend

## Story

Reported (2026-07-21): the user created a codex agent with permission mode
`auto` (expecting a WRITABLE sandbox so it can create files) and asked it to
"create a simple TODO server backend use /flow". The agent refused, reporting:

> I cannot run /flow ... because the filesystem is read-only and approval is
> disabled ... Those writes are blocked.

So an `auto` (and presumably `edit`) codex agent is still running under a
READ-ONLY sandbox. Expected: `auto` -> codex `--sandbox danger-full-access`
(edit + run), `edit` -> `--sandbox workspace-write`, `manual` -> `read-only`.

## Suspected mechanism (verify first, do not assume)

- `_launch_agent_turn` passes `permission_mode=agent.permission_mode` to
  `backend.stream(...)`; `CodexBackend.stream` maps it via `_codex_sandbox_for`
  (manual->read-only, edit->workspace-write, auto->danger-full-access) and calls
  `_stream_app_server(..., sandbox=...)`. Candidate failure points:
  1. The app_server runner (`_stream_app_server` in agent.py) may not actually
     APPLY the `sandbox` argument to the codex app-server `turn/start` (or
     initialize), unlike `codex exec --sandbox`. Probe the app-server protocol:
     does it take a sandbox/approval field per turn, or default to read-only?
  2. The agent's stored `permission_mode` may not be "auto" (did the create/
     settings form persist it? check the record on disk).
  3. codex app-server may ALSO need an approval-policy alongside the sandbox
     (like exec needed `approval_policy=never`), so danger-full-access without
     approvals still blocks writes.

## Repro to build FIRST

- Create a codex agent with `permission_mode="auto"`, chat "create a file
  ./probe.txt", and assert a file was written (or trace the codex app-server
  request to see the sandbox/approval actually sent). A failing repro is the
  first deliverable; then fix the runner to apply the sandbox + any approval
  policy, and pin it.

## Definition of Done

- A codex agent in `auto` can write files; `edit` can edit workspace files;
  `manual` stays read-only (test: the app-server runner receives/sends the mapped
  sandbox + approval; ideally a live probe that a write succeeds under auto).
- Full check suite green.
- manual: an `auto` codex agent can create files.

## Notes
- Do NOT start yet (user's call, 2026-07-21) - filed so it is not forgotten.
- Relevant: scufris/agent.py (`_stream_app_server` - does it apply `sandbox`?),
  scufris/backends.py (`_codex_sandbox_for` / `CodexBackend.stream`),
  scufris/agent_store.py (permission_mode persistence). Lesson
  `probe-runtime-on-target-host-early`: probe the codex app-server sandbox/
  approval wire contract live before designing the fix.
- Also check the claude flavour (claude `--permission-mode` mapping) - whether
  edit/auto actually enable writes there too.
