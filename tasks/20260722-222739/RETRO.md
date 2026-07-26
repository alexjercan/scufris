# Retro: T5 Telegram reply rendering + end-to-end example

- TASK: 20260722-222739
- BRANCH: feat/telegram-t5-rendering
- REVIEW ROUNDS: 1 (APPROVE)

See TASK.md for what/why and REVIEW.md for the finding. This is process only.

## What went well

- Understanding-first paid off: grounding the plan in the exact seams
  (`StreamDone.reply.tool_calls`, the `on_message -> str` contract, the
  `poll_once` test seam already documented in `telegram.py`) meant the feature
  code landed on the first try and mypy/ruff were green immediately.
- Keeping the `OnMessage = Callable[[str], Awaitable[str]]` seam unchanged
  (render inside `on_message`, not a new richer type) kept the blast radius tiny
  and left all existing `on_message` tests passing untouched.
- The out-of-context reviewer earned its keep: it found R1.1 (upfront typing
  action not failure-tolerant, so a transient error drops the update because the
  offset already advanced) - a real robustness gap the implementing session had
  not considered.

## What went wrong

- The first cut of BOTH the e2e test and the example ran the production
  free-running `run()` poll loop under the app lifespan. respx serves getUpdates
  instantly, so the loop never blocked, busy-spun, and the process hung to a 200s
  timeout (killed by PID). Root cause: I reasoned "most realistic = drive the real
  loop", forgetting that a stubbed transport removes the very block a long-poll
  relies on. The deterministic seam (`poll_once` with `run` stubbed) was the
  right call and was already the documented test seam - I should have used it from
  the start. Cost: one hung run + a kill + rewriting two files.
- Early verification runs piped check output through `tail`, which ate the real
  exit code (an existing AGENTS.md rule). Switched to bare runs writing exit codes
  to a file; that is how the single real test failure (missing `sendChatAction`
  stub) surfaced cleanly.

## What to improve next time

- When e2e-testing any poll/retry loop against a mocked-instant transport, drive
  the single-step seam and stub the loop wrapper - do not run the free loop. New
  ledger entry `mock-transport-drive-the-step-not-the-loop`.
- Reach for bare commands with explicit exit-code capture on the FIRST
  verification pass, not after a pipe hides a failure.

## Action items

- [x] Ledger: added `mock-transport-drive-the-step-not-the-loop` (sibling of the
      respx-replies-instantly family).
- No follow-up code tasks: R1.1 was fixed in-cycle; no residual work discovered.
