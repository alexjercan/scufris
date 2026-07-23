# Retro: SC2 orchestrator comms steering

- TASK: 20260723-153615
- DATE: 20260723
- OUTCOME: landed, 1 review round (APPROVE)

## What we set out to do

Teach the orchestrator, via its turn-prompt steering, to close the
`request_input` round-trip on the poll path (auto_wake off by default): at end
of turn call `pending_agents`, answer each blocked sub-agent with
`message_agent`, then `acknowledge`. The complementary half of SC1.

## What went well

- Building on SC1 (already landed on master), the worktree started from a tree
  that already had the two-preamble structure, so this was a focused extension
  of the orchestrator's block with zero conflict.
- The single-block constraint was the one real design decision, and it was
  already understood from SC1: `strip_steering` removes only ONE leading
  sentinel block (`count=1`), so a second sentinel-wrapped comms block would
  survive uncleaned in titles/transcripts. Kept both clauses in one block and
  documented WHY inline, so a future reader does not "tidy" it into two blocks
  and silently break stripping.
- Refactored the giant f-string into named `_HOST_TOOLS_CLAUSE` /
  `_COMMS_CLAUSE` pieces composed into one block - reads far better than
  appending sentences to a wall of concatenated strings, and makes the
  two-clauses-one-block intent obvious.
- Grounded the steering wording against the actual tool docstrings in
  `mcp_server.py` BEFORE writing it, so the names and arg order
  (`message_agent(agent_id, message)`, `acknowledge(agent_id)`) match exactly -
  the reviewer's tool-name check found no drift. Steering that names a
  non-existent tool or wrong arg is worse than no steering.
- Out-of-context review passed round 1 with only a non-blocking nit.

## What went wrong / friction

- Nothing material. The SC1 retro's process lessons (run checks inside
  `nix develop`, confirm pytest green via exit code not the swallowed summary
  line, mypy baseline is pre-existing-red) carried straight over and saved the
  dead commands SC1 spent discovering them - which is the compounding working
  as intended.

## Lessons (candidates for the ledger)

- `orchestrator-steering-is-one-block-two-clauses`: the orchestrator's
  `STEERING_PREAMBLE` must stay a SINGLE `[scufris-tools]` block because
  `strip_steering` strips only the first (`count=1`); add new orchestrator
  guidance as another CLAUSE inside the block, never as a second sentinel block.
- `ground-steering-text-in-the-real-tool-signatures`: before writing steering
  that tells the model to call a tool, read that tool's actual name and
  signature in `mcp_server.py` and match them verbatim - a typo'd name or arg
  steers the model to a call that cannot succeed.
- (carried from SC1, reconfirmed) `scufris-mypy-baseline-is-red`,
  `run-repo-checks-inside-nix-develop`, `nix-develop-pytest-pipe-eats-the-summary`.

## Deferred to Finish

- Manual live-probe DoD: end to end with a real backend - a sub-agent blocks
  (calls `request_input`), and on the next orchestrator turn the orchestrator
  polls `pending_agents`, answers via `message_agent`, and `acknowledge`s it;
  the sub-agent resumes with the answer. Composes with SC1's live probe (they
  are the two halves of the same round-trip, so one probe exercises both).
