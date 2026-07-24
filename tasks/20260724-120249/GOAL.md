# Goal: record session ownership at launch per backend (part 2)

- DATE: 20260724
- UMBRELLA TASK: 20260724-120249
- LANDING SCOPE: squash-merge each task to master (local, no push), as in part 1.

## Goal

Build on the part-1 ownership index (20260724-111947) by recording each
backend's session id / ownership tag at LAUNCH using the backend's strongest
handle, so the index is populated robustly instead of only scraped from
`StreamDone` after the turn. Spike: tasks/20260724-111839/SPIKE.md (part 2).

Scope decision (see 20260724-111955 DECISION.md): deliver the two handles that
are both safe and valuable now -
- **claude**: mint the session UUID in scufris and pass `--session-id <uuid>`, so
  the id is deterministic and known before the turn (not read back from the
  result frame), and a turn that dies before its result frame still has a known
  id via `StreamDone`.
- **opencode**: tag the created session with `metadata={agent_id}` at
  `POST /session`, recording ownership on the provider side.

Explicitly DEFERRED to part 3 (20260724-111959), with rationale in the decision:
the codex per-agent `originator` override (pure risk post-part-1: it would break
the `_SCUFRIS_ORIGINATORS` reads in `read_usage`/health, and listing no longer
depends on it), and `parentID` / `parent_thread_id` hierarchy threading (needs
the parent-agent link part 3 introduces).

## Done means

1. A fresh claude turn passes `--session-id <uuid>` (a scufris-minted UUID), and
   `StreamDone.session_id` equals it (test: `test_claude_stream_mints_session_id`,
   `test_claude_stream_done_carries_minted_id`).
2. A claude turn resuming an on-disk session still uses `--resume` and no
   `--session-id` (test: `test_claude_stream_resumes_existing_session`).
3. An opencode session is created with `metadata` carrying the agent id
   (test: `test_opencode_create_session_tags_agent_metadata`).
4. codex behaviour is unchanged; `_SCUFRIS_ORIGINATORS` reads still work
   (cmd: `grep -n "clientInfo" scufris/agent.py` shows the shared "scufris" name,
   unchanged).

Overall: `nix flake check` green (ruff + mypy + pytest).

## Tasks

- [ ] 20260724-111955 (p40, scufris) Record session ownership at launch per backend

## Decisions (load-bearing, architectural)

- 20260724-111955 DECISION.md: deliver claude+opencode launch handles; defer
  codex originator + parent threading to part 3 (ACCEPTED)

## Manual acceptance (batched for the user at Finish)

- (none expected; all proofs are test:/cmd:)
