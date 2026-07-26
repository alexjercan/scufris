# Retro: Persist codex 'thinking' reasoning across a page reload (backend sidecar)

- TASK: 20260726-215910
- BRANCH: feature/reasoning-sidecar
- REVIEW ROUNDS: 1 (APPROVE, out-of-context)

## What went well

- Reading the existing code before writing paid off twice: `renderChatLog`
  already rendered `entry.reasoning` (built by the dependency task
  20260726-215847), so the frontend work collapsed to plumbing `reasoning`
  through two transcript->ChatMsg mappings - no new rendering code. And the
  capture point fell out of `_launch_agent_turn.turn_stream()`, which already
  watched `StreamSessionStarted`/`StreamDone`.
- Applying the `out-of-context-review-misses-cross-layer-timing` lesson
  up front: put the sidecar write BEFORE the `yield` of the done frame, so a
  reload triggered by `done` reads an already-written sidecar (the on_complete
  persist runs too late). This avoided a timing bug rather than discovering it
  in review.
- The load-bearing design fork (storage shape + alignment) was decided and
  recorded in DECISION.md before coding, so the review had the "why" and there
  was no mid-build reshaping.
- Each layer got a behavioral test at its own boundary (capture via
  `/api/chat/stream`, merge via a real rollout+sidecar, frontend via the real
  `startAgentChat` fetch->render path). The review A/B'd the capture test and
  confirmed it fails with the fix deleted.

## What went wrong

- Planning edits to TASK.md (Flow State, Design) were made in the MAIN checkout
  before sprouting; the worktree cuts from committed HEAD, so those edits were
  not on the branch. Root cause: edited the task record before creating the
  isolated worktree. Cost: had to `git checkout` the main checkout and re-apply
  the edits (plus DECISION.md) inside the worktree.
- The `cd "$(sprout new ...)"` one-liner the flow skill prescribes was denied
  by this harness's EnterWorktree guard (the `sprout new && cd` shape). Root
  cause: the combined create-and-enter is what the guard blocks. Adapted by
  running `sprout new` alone and operating on absolute worktree paths.
- A latent test-isolation defect surfaced: tests that build `Settings()` without
  an explicit `state_dir` write to the real `~/.local/state/scufris`. Harmless
  for the overwrite-based stores, but the new sidecar is APPEND-only, so it grew
  `reasoning/sess-x.json` across runs. Caught by inspecting the real state dir
  after a test run (not by a failing test). Fixed suite-wide with an autouse
  `_isolate_state_dir` conftest fixture - which is exactly the promotion the
  `isolate-state_dir-in-tests-that-assert-config` ledger entry had been
  predicting.

## What to improve next time

- Sprout FIRST, then make all task-record edits (Flow State, plan, DECISION.md)
  inside the worktree - or commit them before sprouting. Edits to the main
  checkout's working tree do not reach a freshly-cut branch.
- When adding a new persisted store, check test isolation as part of the design,
  not after: an append-only store amplifies any "tests hit real state_dir"
  leak. Grep the real state dir (or add the isolation fixture) up front.

## Action items

- [x] Added the autouse `_isolate_state_dir` conftest fixture (promotes the
      x2 `isolate-state_dir-in-tests-that-assert-config` lesson for the
      state_dir half).
- [x] Bumped/annotated the ledger entry; no follow-up code task needed.
- Two review NITs (R1.1 stray `.json.tmp` on failed replace; R1.2 `list[Any]`
  vs a `Protocol` in `merge_reasoning`) left at discretion - both cosmetic and
  consistent with repo conventions.
