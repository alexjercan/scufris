# Decision: attribute sub-agent escalations to the spawning orchestrator chat via SCUFRIS_ORCH_SESSION_ID

- DATE: 20260724-132713
- STATUS: ACCEPTED
- TASK: (planned under umbrella 20260724-132713)
- TAGS: decision, agents, sessions, comms

## Context

Redefines seeded task 20260724-111959. A bare `parent_agent_id` is a no-op: only
the orchestrator has spawn tools (sub-agents get only `request_input`), so the
parent is always the orchestrator, and `pending_agents` is already a global
orchestrator poll. Part 1 gave the orchestrator MULTIPLE chats (sessions), which
creates the real question: when a child calls `request_input`, WHICH orchestrator
chat should see it? Today every chat sees every waiting child.

The mechanism is constrained by the architecture: the `message_agent` /
`run_agent` / `pending_agents` tools run in a SEPARATE MCP subprocess that has no
per-turn session context, and a FRESH orchestrator turn's session id is not known
until the turn completes.

## Decision

Attribute each spawned child to the orchestrator SESSION that spawned it, and
route escalations by it:

1. **Capture the spawning session with zero new threading.** For an orchestrator
   turn, the session id it is resuming is already `session_id`/`thread_id`. Inject
   it into the orchestrator MCP server env as `SCUFRIS_ORCH_SESSION_ID`
   (`scufris_mcp_server`, codex `_mcp_overrides` + claude `_scufris_claude_args`).
   Empty on a fresh turn (no resumed id yet) - the documented edge.
2. **Propagate at spawn.** `message_agent` / `run_agent` read
   `SCUFRIS_ORCH_SESSION_ID` and send `parent_session_id` (parent_agent_id =
   orchestrator) to the child's `/run` / `/chat` endpoint, which records both on
   the child's `SessionRegistry` entry (`parent_agent_id` already reserved there;
   add `parent_session_id`).
3. **Route by it, don't orphan.** `pending_agents` reads the same env and asks
   `GET /api/agents/pending?parent_session_id=X`. The endpoint returns children
   whose `parent_session_id == X` OR is EMPTY (unattributed: UI-spawned, or
   fresh-turn spawns), never a child clearly owned by a DIFFERENT chat, and
   annotates each row with its parent chat. So a chat sees its own children plus
   unattributed ones, and nothing is orphaned.

## Alternatives considered

- **Record `parent_agent_id` only (the literal seeded task)** - no-op today; the
  parent is always the orchestrator and adds no routing signal. Rejected.
- **Annotate-only (show all children, labeled, no filter)** - simpler, but with
  many chats every chat still sees every child; the operator/LLM must
  disambiguate manually. Rejected in favour of filter-with-unattributed-fallback,
  which is the actual routing the user asked for.
- **Hard filter (only exact-match parent_session)** - would ORPHAN UI-spawned and
  fresh-turn children (empty parent) from every chat's poll. Rejected; the
  empty-parent fallback keeps them visible.
- **Thread a fresh turn's minted id in** - a fresh codex turn's id is unknowable
  before completion, and claude/codex would diverge; using the resumed id (empty
  on fresh) is uniform and the edge is acceptable.

## Consequences

- With multiple orchestrator chats, a child's `request_input` surfaces to the
  chat that spawned it; unattributed children stay visible everywhere (no
  regression to the single-chat flow, where parent is empty and everything shows).
- New env var `SCUFRIS_ORCH_SESSION_ID` (orchestrator role only) and a
  `parent_session_id` field on the registry entry + the pending API row.
- Fresh-turn edge: a child spawned in a brand-new chat's first turn is
  unattributed until that turn finishes; documented, not a bug.
