# Decision: the session registry owns per-agent session history; the switcher lists from it, not from a disk scan

- DATE: 20260724-111947
- STATUS: ACCEPTED
- TASK: 20260724-111947
- TAGS: decision, agents, sessions, backend

## Context

The orchestrator's session switcher (`GET /api/agent/sessions`) is built by
`list_sessions()` (`sessions.py`), which scans codex rollout files on disk and
keeps those whose `originator in {"codex_exec","scufris"}` AND `cwd ==
os.getcwd()`. Every scufris-driven codex turn - orchestrator OR sub-agent -
initializes app-server with `clientInfo.name = "scufris"` (`agent.py:515`), so
originator cannot distinguish them, and a sub-agent bound to the server dir
shares the cwd. Result: a codex sub-agent's chat leaks into the orchestrator's
switcher. `SessionRegistry` (`agent_store.py`, `sessions.json`) already records
the true owner of the CURRENT session per agent, but not the agent's session
HISTORY, so the switcher falls back to the leaky scan. See
`tasks/20260724-111839/SPIKE.md` for the full analysis and the rejected
own-the-transcript alternative.

## Decision

Make `SessionRegistry` the single source of truth for session OWNERSHIP and
history. Each entry grows from `{backend, session_id}` to
`{backend, session_id (current), sessions: [id,...], parent_agent_id: str|None}`
(`parent_agent_id` is reserved here for part 3, populated later). Ownership is
RECORDED as sessions are minted/switched, never inferred from the store:

- `mark_finished` APPENDS each newly-minted session id to the owning agent's
  history (was: overwrite current only).
- "New chat" (`set_orchestrator_session(None)`) sets current to None but KEEPS
  the history; the next turn's fresh id is appended by `mark_finished`.
- "Switch" records the target id as current, appending it to history if unseen.
- Deleting a session removes that id from the owner's history.
- A backend switch still clears the whole entry (sessions are backend-specific:
  a codex id is meaningless to claude), starting a fresh history.

`GET /api/agent/sessions` is rewritten to list the ids the registry attributes
to `ORCHESTRATOR_ID` under its current backend, hydrating each id's title/times
through the agent's backend (`read_transcript` for the first user message,
`read_status` for the mtime) - so it is backend-agnostic and gives claude and
opencode multi-session for free. `list_sessions()` is no longer used by the
switcher (it stays only for `health.py`'s rough diagnostic count).

**Forward-only: no backfill.** Sessions that predate registry-history tracking
are NOT imported into the switcher. Backfilling would have to re-scan disk with
the exact `(originator, cwd)` heuristic that is broken, re-importing sub-agent
chats into the orchestrator's persistent history - reintroducing the bug into
the store. Old rollouts stay on disk (nothing is deleted); they just do not
appear in the switcher. The orchestrator's CURRENT chat is already registry-
tracked, so no in-flight conversation is lost.

## Alternatives considered

- **Narrow the disk-scan filter (e.g. a per-agent codex originator)** - the
  `20260720-020345` direction. Patches only codex, only the cwd collision,
  stays inference-based, and gives claude/opencode no multi-session. Leaves the
  design defect (ownership guessed, not recorded) in place. Rejected; this
  decision supersedes that task.
- **scufris owns the full transcript, re-inject each turn** - rejected in the
  spike: lossy for these CLIs (drops native tool-call/reasoning events) and
  breaks prompt caching. See SPIKE.md option B.
- **A sibling session-records store instead of extending the registry** -
  viable, but splits session facts across two files against the spirit of
  `tasks/20260723-001251/DECISION.md` (one home for session ids). Folding into
  the registry keeps a single owner.
- **Auto-backfill old chats** - rejected above (re-imports the leak).

## Consequences

- The leak is structurally impossible: the switcher only ever shows ids the
  registry attributes to the orchestrator.
- claude and opencode gain multi-session listing with no backend-specific code.
- Hydrating each id via `read_transcript`/`read_status` reads more of each
  rollout than `list_sessions`' head-only read; acceptable for a short switcher
  list, but a large history is O(sessions * rollout size) IO. If it bites, cache
  the per-id head or add a cheap `session_info` head-read per backend.
- git_branch / cwd on `SessionInfo` become best-effort (None off codex), since
  the generic hydration path does not parse `session_meta`. The switcher UI does
  not depend on them today.
- `sessions.json` schema changes; the loader must tolerate the legacy
  `{backend, session_id}` shape (treat as `sessions=[session_id]`).
- Old pre-tracking chats drop out of the switcher (forward-only); documented as
  expected. They remain on disk.
