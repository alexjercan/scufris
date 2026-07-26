# Review: Persist codex 'thinking' reasoning across a page reload (backend sidecar)

- TASK: 20260726-215910
- BRANCH: feature/reasoning-sidecar

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

The branch delivers the spec cleanly across all three layers (capture in
`app.py`, merge in `sessions.py`/`backends.py`, frontend re-hydration), each
pinned by a meaningful behavioral test. No BLOCKER or MAJOR findings.

Checks run in the worktree (nix devshell), all PASS:
- `ruff check .` -> All checks passed
- `mypy .` -> no issues in 54 source files
- `python -m pytest -p no:warnings` -> 548 passed
- Frontend `npx vitest run` -> 184 passed (19 files); `tsc --noEmit` PASS;
  `npm run lint` PASS

In-session re-verification of a load-bearing claim (per the review skill): A/B'd
`test_chat_stream_captures_reasoning_to_the_sidecar` by deleting the capture
block in `app.py` `turn_stream()`; the test FAILED (sidecar empty: "Right
contains one more item: 'let me think'"), then `git checkout` restored the
committed file cleanly. The capture path is genuinely pinned.

Verified correct (not findings): session-id path-traversal guard
(`test_unsafe_session_id_is_a_noop_not_a_traversal`); tail-alignment under the
200-message transcript cap (`zip(strict=False)` stops at the shorter sequence);
streamed-vs-on-disk answer whitespace normalization in `reasoning_fingerprint`;
partial/drifted sidecar degrades via the fingerprint-guard break; capture happens
before the done frame is yielded; codex-gated capture.

- [ ] R1.1 (NIT) scufris/reasoning_store.py:114-119 - `_persist` leaves the
  `.json.tmp` file behind if `os.replace` fails (no `finally` cleanup). Cosmetic
  and consistent with the sibling stores' pattern; no change required.
  - Response:
- [ ] R1.2 (NIT) scufris/sessions.py:527 - `merge_reasoning(entries: list[Any])`
  is typed loosely to avoid a `reasoning_store -> sessions` import cycle. A
  `Protocol` with `answer: str`/`reasoning: str` would document the contract;
  optional.
  - Response:

### Pending manual DoD (user acceptance, not resolved by review)

- After a FULL page reload, past assistant turns show the collapsed "thinking"
  spoiler expandable to the reasoning that streamed (send a codex turn, reload,
  expand).
- The spoiler is collapsed by default on reload (`<details>` closed on first
  render).
- Pre-existing sessions without a sidecar render normally: no reasoning, no
  error.
