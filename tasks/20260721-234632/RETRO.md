# Retro: U4 - orchestrator-at-root settings symmetry + global sections

- TASK: 20260721-234632
- BRANCH: feature/settings-root-symmetry (landed 7f96905)
- REVIEW ROUNDS: 2 (out-of-context REQUEST_CHANGES on a MAJOR, then in-session APPROVE)

See TASK.md for what/why and REVIEW.md for the findings. Process notes only here.

## What went well

- The consolidation was clean where it mattered: one shared `agentSettingsDeps`
  builder drives BOTH `/settings` and `/agents/orchestrator/settings`, so they are
  provably the same component with the same data; backend/model live once (the
  agent form); the ~250-line dead composition + its helpers came out with no
  dangling refs. The reviewer confirmed the symmetry, reload thread, and sweep.

## What went wrong

- R1 (MAJOR): retiring `settings-view.renderSettings` silently DROPPED its
  read-only-server handling (`const live = config.writable && actions`), and I
  did not re-wire it in the unified component - so global write controls rendered
  live and 403 on a read-only server. Worse, I had WRITTEN in TASK.md's notes that
  "the read-only path gets wired here (config.writable)" when it was not. Two
  faults: (a) `moving-logic-off-a-scope-drops-its-incidental-guarantees` - the
  old render's read-only branch was an incidental guarantee I forgot to
  re-establish; (b) an honesty gap - I claimed a DoD item done before it was.
- Root cause of (a): I focused on the happy-path composition and the
  no-duplication design, and did not enumerate what the surface I was REPLACING
  provided (its writable gate) before deleting it.

## What to improve next time

- When RETIRING a render/surface, list what it did that the replacement must also
  do (here: the `config.writable` gate + the read-only note) and re-establish each
  BEFORE deleting - the same enumerate-the-scope discipline as moving logic off a
  lock.
- Never write "X is wired/done" in a TASK note until the code path exists and is
  tested; a claimed-but-absent DoD item is worse than an open one.

## Action items

- [x] Adopted R1 (writable now per-load from config.writable, gates form + global
      sections + a read-only note), R2 (read-only tests hit the real data path),
      R3 (dead .settings__panels CSS removed).
- [x] Bumped `moving-logic-off-a-scope-drops-its-incidental-guarantees` to x2.
