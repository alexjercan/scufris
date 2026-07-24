# Record session ownership at launch per backend (claude --session-id, opencode metadata+parentID, codex originator/parent read-back)

- STATUS: OPEN
- PRIORITY: 40
- TAGS: spike, agents, sessions

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

## Notes

- Spike: tasks/20260724-111839/SPIKE.md (part 2)
- Depends on part 1 (tatr 20260724-111947) landing the index first.
- Verify claude `--session-id` uniqueness behaviour and that a per-agent codex
  originator does not break `_SCUFRIS_ORIGINATORS` reads (`read_usage`, health).

