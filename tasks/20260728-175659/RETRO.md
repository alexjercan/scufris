# Retro: Telegram /cancel stops current orchestrator message

- TASK: 20260728-175659
- BRANCH: feature/telegram-cancel-command
- REVIEW ROUNDS: 1

## What went well

- Reusing the existing supervisor cancel path kept the app wiring small and
  avoided duplicating the HTTP cancel endpoint inside the Telegram transport.
- Diff review caught the important scheduling bug before close, and the fix got
  a regression test that models `/cancel` arriving on the next poll.

## What went wrong

- The first implementation treated command dispatch as the whole problem and
  missed that `_dispatch` awaited `_render_turn`, blocking the long-poll loop
  from receiving a later `/cancel` until after the turn completed.
- The sandbox could not run the real project gate because the toolchain is only
  available through Nix and the Nix daemon socket is blocked here.

## What to improve next time

- For transport commands that affect in-flight work, review whether the receive
  loop can still receive the command while work is active before writing code.
- When a task changes a constructor in a central transport class, sweep all call
  sites immediately and update test helpers before running checks.

## Action items

- [x] Captured the transport scheduling lesson in the task close-out.
