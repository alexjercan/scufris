# Spike: session ownership index for the multi-agent orchestrator

- DATE: 20260724-111839
- STATUS: RECOMMENDED
- TAGS: spike, agents, sessions, backend

## Question

Sub-agent provider sessions leak into the orchestrator's session switcher: a
`codex` sub-agent bound to the server's own directory shows up in the
orchestrator's list of chats. How should scufris manage sessions so that (1)
each agent's session list is correctly scoped to that agent, (2) a sub-agent
carries a reference to the session/agent that spawned it (for the
`request_input` escalation loop), and (3) the whole thing generalizes across the
codex, claude, and opencode backends?

A secondary, deliberately-considered option: should scufris stop relying on the
provider session stores at all and instead OWN the full conversation transcript,
re-injecting it into codex/claude/opencode each turn as a "this is your history"
payload?

A good answer picks a concrete storage/ownership model, says why it beats the
alternatives (especially the own-the-transcript option), and is concrete enough
that `/plan` can expand it into steps without re-litigating the choice.

## Context

- **The leak is one function.** `list_sessions()` in `scufris/sessions.py:269`
  (called by `GET /api/agent/sessions`, `app.py:1650`) reconstructs the
  orchestrator's session list by scanning codex rollout files on disk, filtered
  by `originator in {"codex_exec", "scufris"}` AND `cwd == os.getcwd()`. Every
  scufris-driven codex turn calls `initialize` with `clientInfo.name = "scufris"`
  (`agent.py:515`), so orchestrator and sub-agent rollouts are indistinguishable
  by originator; and a sub-agent whose project is bound to the server dir shares
  the cwd. So the filter cannot separate them. Ownership is INFERRED from the
  store, and the inference is wrong.

- **scufris already records the truth - partially.** `SessionRegistry`
  (`agent_store.py:104`, `<state_dir>/sessions.json`) maps
  `agent_id -> {backend, session_id}` for every agent, orchestrator included
  (see `tasks/20260723-001251/DECISION.md`). But it stores only the CURRENT
  session per agent. Multi-session history (the orchestrator's several chats) is
  NOT tracked, which is exactly why the switcher falls back to a disk scan and
  inherits the collision.

- **Transcripts are provider-owned and rich.** Each backend records native
  tool-call and reasoning events in its own store (codex rollout JSONL, claude
  `~/.claude/projects/<cwd-slug>/<id>.jsonl`, opencode over its HTTP daemon).
  scufris reads these back by id (`read_transcript`/`read_context`) to
  re-render history. A codex fork already flattens prior turns to text
  (`format_fork_seed`, `sessions.py:517`) - the only place scufris re-injects
  history today.

- **Escalation already works, loosely.** A sub-agent signals via `request_input`
  (`mcp_server.py:629`) using `SCUFRIS_AGENT_ID` from its env; the orchestrator
  polls `pending_agents` and answers with `message_agent`, which resumes the
  sub-agent's session. There is no explicit parent link - the orchestrator is
  the implicit parent of everything.

### Backend capabilities (from research, 2026-07-24; codex `rust-v0.146.0-alpha.6`,
Claude Code v2.1.x, opencode v1.18.4)

| Capability | codex (app-server) | claude (`claude -p`) | opencode (`serve`) |
| --- | --- | --- | --- |
| Set our own session id | No (server UUID) | **Yes** `--session-id <uuid>` | No (server id/slug) |
| Persist an arbitrary tag | Partial: `originator` (+ role/nickname) in `session_meta`; no free-form metadata | No | **Yes** `metadata: {}` at `POST /session` / `PATCH` |
| Native parent link | **Yes** `parent_thread_id`, `forked_from_id` in `session_meta` | No (sub-agents share one session) | **Yes** `parentID`; `GET /session/:id/children` |
| Server-side list/filter | Scan `CODEX_HOME/sessions/YYYY/MM/DD/`; `resume` filters by cwd | Client-side scan of the project dir | **Yes** `GET /session` by `directory`, `parentID`, ... |

Sources: openai/codex `codex-rs/rollout/src/recorder.rs`,
`codex-rs/protocol/src/protocol.rs`,
`codex-rs/app-server-protocol/src/protocol/v2/thread.rs`;
code.claude.com/docs/en/{cli-reference,sessions,headless,sub-agents};
anomalyco/opencode `packages/opencode/src/session/session.ts`,
`.../server/routes/instance/httpapi/groups/session.ts`, `.../tool/task.ts`.

Cross-framework prior art on transcript ownership: OpenHands (append-only
`EventLog` it owns; derived view never persisted), Aider (assembles + sends
`all_messages()` each turn, `ChatSummary` compacts, manages cache-control),
AutoGen/AG2 (`chat_messages: Dict[Agent, List]` - native per-partner history),
LangGraph (checkpointer + `thread_id`; `interrupt`/`Command(resume=...)` for
human-in-the-loop), crewAI (own memory store, per-task fact injection). All own
NATIVE message structures, and only because they talk to STATELESS chat APIs;
the coding CLIs are stateful, so appending-by-resume already gives us what they
rebuild by hand. Prompt caching (Anthropic/OpenAI) survives resume-append but is
broken by re-seeding/compaction that rewrites the prefix.

## Options considered

- **A. Provider store + resume-by-id (status quo).** Rely on the CLI's own
  session store; discover the list by scanning + filtering. Pros: zero extra
  storage; transcripts keep native tool/reasoning fidelity; prompt caching for
  free. Cons: ownership is inferred, not recorded - which is the bug; the
  `(originator, cwd)` filter is unfixable in general (originator is a client
  TYPE, cwd is shared); no multi-session for claude/opencode.

- **B. scufris owns the full transcript, re-inject each turn ("this is your
  history").** Generalize `format_fork_seed` to every backend and every turn;
  start a fresh provider session each turn seeded with the whole conversation.
  Pros: total independence from provider stores; uniform across backends;
  trivially correct ownership. Cons - decisive against: (1) LOSSY for these
  CLIs specifically - the rollouts hold native tool-call and reasoning events
  that have no faithful plain-text re-injection (codex's own history-resume path
  is even marked "FOR CODEX CLOUD - DO NOT USE"); (2) BREAKS prompt caching -
  re-seeding re-pays input tokens every turn and cost grows quadratically, where
  resume-append keeps the cached prefix; (3) reinvents compaction/summarization
  that the CLIs already do internally. Every framework that owns its transcript
  keeps NATIVE messages and still only does it because its API is stateless -
  which ours is not.

- **C. Hybrid: provider transcripts (A) + a scufris-owned OWNERSHIP INDEX.**
  Keep transcripts in the provider stores (A's fidelity + caching), but stop
  INFERRING ownership: record it in a scufris-owned index keyed by an id scufris
  controls, and use each backend's strongest handle to make the record robust at
  launch. The switcher lists ids the index attributes to the agent, then
  hydrates title/context per id from the store. Pros: kills the leak
  structurally (ownership recorded, not guessed); gives claude/opencode
  multi-session for free; models hierarchy explicitly. Cons: more moving parts
  than A; per-backend handle work.

- **D. Do nothing / narrow the filter.** E.g. also filter codex sessions by a
  per-agent originator only. Pros: tiny. Cons: only patches codex, only the cwd
  collision, still inference-based, still no claude/opencode multi-session -
  leaves the design defect in place.

## Recommendation

**Option C, the hybrid.** Keep provider-owned transcripts; add a thin
scufris-owned ownership index; record ownership at launch using each backend's
strongest handle. Reject B: for these CLIs it trades away tool/reasoning
fidelity and prompt caching to solve a problem (ownership) that an index solves
without those costs. Reject D: it patches a symptom and leaves the
inference-based design that will keep generating these bugs.

Concrete shape for scufris, in three parts:

1. **Index owns ownership + multi-session history.** Extend `SessionRegistry`
   (or add a sibling session table) from `agent_id -> {backend, current}` to
   also carry the full set of sessions an agent has owned and each session's
   `parent_agent_id`. `mark_finished` already sees each new session id as it is
   minted - have it APPEND to the agent's list, not just overwrite `current`.
   Rewrite `GET /api/agent/sessions` to list the ids the index attributes to
   `ORCHESTRATOR_ID` and hydrate each title/context by id via the existing
   `read_transcript`/`read_context`. Retire `list_sessions`' disk-scan +
   `(originator, cwd)` filter. This alone fixes the reported bug and generalizes
   multi-session to claude/opencode.

2. **Record ownership at launch via each backend's handle.** claude: generate
   the UUID in scufris and pass `--session-id <uuid>` (deterministic filename,
   id known before the turn instead of scraped from `StreamDone`). opencode:
   `POST /session` with `metadata={agent_id}` and the real `parentID`; filter
   server-side. codex: cannot set id and has no free-form metadata, but with the
   index authoritative it does not need to - optionally set a per-agent
   `originator` for defense in depth and read back `parent_thread_id`/
   `forked_from_id` from `session_meta` to corroborate hierarchy.

3. **Thread the parent reference for escalation.** Inject `parent_agent_id`
   into the child's launch context alongside `SCUFRIS_AGENT_ID`, and keep the
   `request_input` -> `pending_agents` -> `message_agent` loop as the escalation
   channel (it matches LangGraph's interrupt/resume and AutoGen's user-proxy
   shape; no CLI offers native child->parent callback). This makes "which
   session spawned me" explicit rather than "the orchestrator, implicitly".

## Open questions

- **Index vs. sibling table.** Fold multi-session + parent into
  `SessionRegistry`'s JSON, or add a dedicated session-records store? Leaning
  fold-in to keep one home for session facts (the `20260723-001251` decision's
  spirit); `/plan` should settle the exact schema.
- **Backfill.** Existing orchestrator rollouts predate the index. Do we
  one-time-migrate them by scanning disk (the last legitimate use of the old
  filter), or just start recording forward and let old chats age out? Probably
  a bounded one-time backfill for the orchestrator id only.
- **claude `--session-id` uniqueness.** UUID validity is documented; per-store
  uniqueness enforcement is not - verify a re-used id does not clobber before
  relying on it.
- **codex per-agent originator** would change the recorded `originator`; confirm
  it does not break `_SCUFRIS_ORIGINATORS`-based reads elsewhere (it should not,
  once listing is index-driven, but check `read_usage`/health).

## Next steps

Direction-level tasks this spike seeded, for `/plan` to break into steps:

- tatr 20260724-111947: index-owned session ownership + multi-session history; drive the switcher from it (part 1)
- tatr 20260724-111955: record session ownership at launch per backend (claude --session-id, opencode metadata+parentID, codex originator/parent read-back) (part 2)
- tatr 20260724-111959: thread parent_agent_id into child launch context for the request_input escalation loop (part 3)

## Fix record

(none yet)
