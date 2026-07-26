# DECISION: reasoning sidecar storage + alignment

- STATUS: ACCEPTED
- DATE: 2026-07-27

## Context

Codex reasoning ("thinking") plaintext exists only in the live
`reasoning_delta` stream; the on-disk rollout stores it as an encrypted blob.
To survive a hard page reload, scufris must capture the live stream into its
own sidecar and merge it back into the `/transcript` response at read time.
Two load-bearing shape choices had no single forced answer, so they are
recorded here.

## Decision 1: where the sidecar lives, and its file layout

Store the sidecar in scufris's own `state_dir` (the existing
`~/.local/state/scufris`), as ONE JSON file per session:
`state_dir/reasoning/<session_id>.json`.

Alternatives considered:

- Beside codex's rollout under `codex_home` (the task's "e.g. alongside the
  transcript" hint). Rejected: makes scufris write into codex's private state
  dir, and couples the sidecar's lifetime to codex's file layout.
- One shared `reasoning.json` keyed by session id (mirrors `sessions.json`,
  `agents.json`). Rejected: those stores hold one small row per agent; a
  reasoning store holds one growing list per session per turn, and the atomic
  full-file rewrite would rewrite EVERY session's reasoning on every turn
  (O(all sessions) per append). Per-session files keep an append O(1).

The per-session file mirrors the other stores' write discipline (atomic
tmp+replace, tolerant load) but not their single-file shape.

## Decision 2: how sidecar entries align to transcript assistant messages

Positional tail-alignment guarded by a normalized text fingerprint:

- The store holds one entry per COMPLETED assistant turn, oldest->newest,
  matching the order `read_transcript` surfaces assistant messages. Turns with
  no reasoning still get an entry (empty string) so the sequences stay 1:1.
- At merge time, walk the assistant messages and sidecar entries from the tail
  (newest) backwards, pairing them; each pair is accepted only if a normalized
  fingerprint of the answer text matches. On mismatch, stop attaching to older
  messages.

Alternative considered: correlate codex's stream item-ids to the rollout's
on-disk item-ids for exact keying. Rejected: the rollout reasoning item is the
encrypted one (not the agent_message), the id correspondence is undocumented
and unverified, and it is more fragile and more code than positional
alignment.

Why the fingerprint guard: it makes a PARTIAL or pre-existing sidecar (feature
deployed mid-session, or a turn scufris never captured because it was not
running) degrade gracefully - reasoning attaches only to the turns it genuinely
covers, and a gross mismatch yields no reasoning rather than a mislabeled
spoiler. This directly serves the "existing sessions degrade gracefully, no
error" Definition of Done.
