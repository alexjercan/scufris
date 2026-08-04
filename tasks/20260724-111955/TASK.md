# Record session ownership at launch per backend (claude --session-id, opencode metadata+parentID, codex originator/parent read-back)

- PRIORITY: 40
- TAGS: agents, sessions, backend
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

Record session ownership at LAUNCH using each backend's strongest handle, so
the index (part 1) is populated robustly rather than scraped after the fact.
- claude: generate the session UUID in scufris and pass `--session-id <uuid>`
  (`backends._claude_stream_args`); the filename becomes deterministic and the
  id is known before the turn instead of read back from `StreamDone`.
- opencode: create the session via `POST /session` with `metadata={agent_id}`
  and the real `parentID`; filter server-side by directory/children.
- codex: cannot set the id or free-form metadata; optionally set a per-agent
  `originator` (`clientInfo.name` / `CODEX_INTERNAL_ORIGINATOR_OVERRIDE`) for
  defense in depth and read back `parent_thread_id`/`forked_from_id` from
  `session_meta` to corroborate hierarchy.

Scope (see DECISION.md): part 1 already records ownership after the turn, so this
is a robustness upgrade. Deliver the two safe+valuable handles now (claude
`--session-id`, opencode `metadata`); DEFER the codex per-agent `originator`
(pure regression risk vs `_SCUFRIS_ORIGINATORS` reads, zero payoff post-part-1)
and `parentID`/`parent_thread_id` hierarchy (needs part 3's parent link) to part
3 (20260724-111959).

## Steps

- [x] claude: in `ClaudeBackend.stream` (`backends.py`), mint a UUID
      (`uuid.uuid4()`) for a FRESH turn and thread it into `_claude_stream_args`.
      Change `_claude_stream_args` so: an on-disk `session_id` -> `--resume
      session_id` (unchanged); otherwise -> `--session-id <fresh-uuid>` (never
      reuse a stale/foreign id as `--session-id`, since claude requires a valid
      UUID and a codex id would be wrong/invalid). Keep the scufris MCP flags on
      every turn.
- [x] claude: in `ClaudeBackend.stream`, guarantee `StreamDone` carries the id -
      if the parsed `StreamDone.session_id` is None (e.g. a result frame without
      it), substitute the minted id. Leave `StreamError` as-is (a fresh session
      that errored before producing anything has nothing to resume).
- [x] opencode: add an optional `metadata: dict[str, Any] | None` to
      `OpencodeClient.create_session` (`opencode_client.py`), included in the
      `POST /session` body when present. In `OpenCodeBackend._send`
      (`backends.py`) pass `metadata={"agent_id": agent_id}` when creating a
      session (thread `agent_id` from `stream` into `_send`).
- [x] Tests (`tests/test_backends.py`): a fresh claude turn's argv contains
      `--session-id <uuid>` and no `--resume`; a resume turn contains `--resume`
      and no `--session-id`; `StreamDone` carries the minted id when the result
      frame omits it. opencode (`tests/test_opencode_backend.py` /
      `test_opencode_client.py`): `create_session(metadata=...)` puts it in the
      POST body, and a fresh `_send` tags `{"agent_id": ...}`.
- [x] codex: confirm UNCHANGED - `clientInfo.name` stays the shared "scufris"
      (no per-agent originator), so `_SCUFRIS_ORIGINATORS` reads keep working.
- [x] NOTES.md fix/design record.

## Definition of Done

- A fresh claude turn passes a scufris-minted `--session-id <uuid>` and
  `StreamDone.session_id` equals it (test: `test_claude_stream_mints_session_id`,
  `test_claude_stream_done_carries_minted_id`).
- A claude turn resuming an on-disk session uses `--resume` and NOT
  `--session-id` (test: `test_claude_stream_resumes_existing_session`).
- An opencode session is created with `metadata` carrying the agent id
  (test: `test_opencode_create_session_tags_agent_metadata`).
- codex `clientInfo` is unchanged (cmd: `grep -n "\"name\": \"scufris\"" scufris/agent.py`).
- Full QA gate green (cmd: `nix flake check`).

## Notes

- Spike: tasks/20260724-111839/SPIKE.md (part 2). Decision:
  tasks/20260724-111955/DECISION.md. Umbrella: 20260724-120249.
- Depends on part 1 (tatr 20260724-111947) - LANDED (236c129).
- Relevant files: `scufris/backends.py` (`ClaudeBackend.stream`,
  `_claude_stream_args`, `_find_claude_session`, `OpenCodeBackend._send`),
  `scufris/opencode_client.py` (`create_session`), `scufris/agent.py`
  (`clientInfo`, left unchanged).
- claude `--session-id` must be a valid UUID; mint with `uuid.uuid4()`. Do not
  feed a resumed/stale id into `--session-id`.


## Outcome (CLOSED)

Shipped on `fix/session-launch-handles`. claude fresh turns now pass a
scufris-minted `--session-id` (deterministic, known before the turn, always
carried on `StreamDone`); opencode fresh sessions are tagged with
`metadata={"agent_id": ...}`. codex is untouched, so `_SCUFRIS_ORIGINATORS`
reads keep working. See NOTES.md (design/fix) and DECISION.md (scope: codex
originator dropped, parent threading deferred to part 3).

- All Steps landed; every DoD proof passes; 429 pytest + ruff + mypy +
  `nix flake check` green.
- Tests written first (red for the right reason), plus an A/B on the StreamDone
  substitution (red when neutered).
- Scope narrower than the spike's part-2 bullets, by design (DECISION.md); the
  remainder is carried into part 3 (20260724-111959), not dropped.
