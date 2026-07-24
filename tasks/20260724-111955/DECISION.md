# Decision: part 2 delivers the claude + opencode launch handles; codex originator and parent threading defer to part 3

- DATE: 20260724-120249
- STATUS: ACCEPTED
- TASK: 20260724-111955
- TAGS: decision, agents, sessions, backend

## Context

The spike (tasks/20260724-111839/SPIKE.md) part 2 proposed recording session
ownership at launch via each backend's strongest handle: claude `--session-id`,
opencode `metadata`+`parentID`, and (optional) a codex per-agent `originator`
plus reading back `parent_thread_id`/`forked_from_id`. Part 1
(20260724-111947) has since landed: the switcher and all ownership are driven by
the scufris-owned registry, and each session id is already captured after the
turn via `mark_finished(session_id=...)`. So "record at launch" is now a
ROBUSTNESS upgrade over a working baseline, not a correctness fix - which changes
the value/risk of each proposed handle.

## Decision

Part 2 implements the two handles that are both safe and valuable given part 1:

- **claude `--session-id <uuid>`**: `ClaudeBackend.stream` mints a UUID for a
  fresh turn and passes it, so the id is deterministic, known before the turn,
  and still carried on `StreamDone` even if the result frame is missing (a turn
  that dies mid-way no longer loses its session id). A resume still uses
  `--resume`, never `--session-id`.
- **opencode `metadata={agent_id}`**: `create_session` tags the new session with
  the owning agent id, recording ownership on the provider side.

DEFERRED to part 3 (20260724-111959):

- **codex per-agent `originator`**: dropped. It is marked "optional" in the task,
  and post-part-1 it carries pure risk with no benefit: `read_usage`/health and
  `sessions.list_sessions` filter on `originator in _SCUFRIS_ORIGINATORS`
  (`{"codex_exec","scufris"}`), so a per-agent originator would silently drop
  scufris's own sessions from those reads. Listing no longer depends on
  originator at all, so there is nothing to gain.
- **`parentID` / `parent_thread_id` hierarchy**: needs the `parent_agent_id`
  link that part 3 introduces; there is no parent source to thread yet.

## Alternatives considered

- **Implement all three handles now** - rejected: the codex originator change is
  a live regression risk against `_SCUFRIS_ORIGINATORS` reads for zero current
  payoff, and parentID has no data to carry until part 3.
- **Thread a store-minted claude UUID from the supervisor through `stream`** -
  heavier cross-cutting change; minting inside `ClaudeBackend.stream` is
  self-contained and achieves the same "id known at launch, carried on
  StreamDone" guarantee. The backend IS scufris code, so "minted in scufris"
  holds.

## Consequences

- claude sessions get deterministic, pre-known ids; a mid-turn failure no longer
  orphans the session. opencode sessions carry an ownership tag for future
  server-side filtering / part 3.
- codex is untouched, so all `_SCUFRIS_ORIGINATORS` reads keep working.
- Part 2's scope is narrower than the spike's part-2 bullet list; the remainder
  is explicitly carried into part 3, not silently dropped (recorded here and in
  GOAL.md).
