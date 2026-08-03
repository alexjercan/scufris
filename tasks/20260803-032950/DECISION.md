# Decision: Source the health session summary from the backend adapter

- DATE: 20260803-140000
- STATUS: ACCEPTED
- TASK: 20260803-032950
- TAGS: agents, backend, frontend

## Context

`scufris/health.py:253-262` fills `AgentHealth.session_count` and `last_session`
with `list_sessions(resolve_codex_home(settings), os.getcwd())` whenever
`settings.agent_enabled`, whatever backend is being probed. Reproduced on
`master`: one scufris-originated rollout under `codex_home` makes
`agent_health(settings, backend=...)` report `session_count == 1` and the same
`last_session` for codex, claude, opencode AND mock.

It is the last codex-shaped reader outside `scufris/backends/` -
`20260801-100415`'s close-out named it - and it survives because both the legacy
and scoped health routes carry it, so they agree and its contract test passes.

What already exists:

| Surface | Backend-correct? | How |
|---|---|---|
| `AgentBackend.read_memory_footprint` | yes | `Capability[MemoryFootprint]`: codex reads its rollouts, claude/opencode/mock answer unsupported |
| `/api/agent/sessions` switcher | yes | ownership registry + `session_info` per id |
| Memory panel (`web/src/agent-settings-panels.ts:188`) | yes | renders the footprint's `session_count` |
| Health card session line | NO | the codex disk scan above |

`agent_health(settings, *, backend, ...)` takes `Settings` only - no registry,
no store handle - and its three renderers (web Health card, Telegram
`render_health`, raw JSON) read `session_count` as a flat int.

## Decision

1. **The summary comes from `read_memory_footprint`.** `health.py` asks
   `get_backend(effective_backend)`; supported -> the footprint's
   `session_count` and `newest`, unsupported -> no reading. No new protocol
   method, and `health.py` loses its `sessions` imports.
2. **`session_count` becomes `int | None`.** `None` = no reading was taken (the
   backend has no session reader, or the agent is disabled); a number is a real
   reading, `0` included. Renderers omit the session line/bit on `None`.
3. **The codex counting scope changes, and that is accepted.** From "rollouts
   with a scufris originator in the server's cwd" to "every `rollout-*.jsonl`
   under `codex_home`".

## Alternatives considered

- **Add `list_sessions` to the `AgentBackend` protocol.** A second
  session-inventory method beside `read_memory_footprint`, plus claude and
  opencode listers nothing in this task requires. Rejected: YAGNI - health needs
  a count and a timestamp, not a list.
- **Branch on `effective_backend == "codex"` in `health.py`.** Smallest diff and
  it preserves the codex scope exactly, but it keeps a codex-only session reader
  outside `scufris/backends/`, so a fifth adapter must find this call site -
  what the epic's backend-first diagnostics removed everywhere else.
- **Drop `session_count`/`last_session` from `AgentHealth`.** The switcher and
  the Memory panel already carry the information. Rejected: the Story asks for a
  backend-correct count on this surface, not for its removal.
- **Count the ownership registry** (`SessionRegistry.sessions_for`).
  Backend-neutral and cheap to count, but `last_session` needs a `read_status`
  per session (a transcript/rollout parse each) on every health probe, and
  `agent_health` has no registry handle - a new dependency for one line.
- **Keep `session_count: int` and report `0` when unsupported.** Rejected: it is
  the zero-that-reads-as-a-measurement the `Capability` docstring warns about.
- **A `Capability[MemoryFootprint]` field on `AgentHealth`.** The envelope's
  third state (supported, value `None`) cannot occur here - a supported reader
  always returns a count - so it would add an unwrap to three renderers for no
  distinction gained.

## Consequences

- A claude/opencode/mock health card shows no session line; the operator gets
  the count from the switcher or the Memory panel, both backend-correct.
- The codex number changes on upgrade: it now matches the Memory panel's
  `session_count` for the same agent instead of disagreeing with it, and
  includes interactive TUI sessions and other directories.
- `web/src/agent-types.ts` widens `session_count` to `number | null`; every
  consumer of `AgentHealth` must handle null (two renderers today).
- A disabled agent reports `None` rather than `0`, since no reading is taken.
- A future backend that grows a footprint reader gets the health count for free.
