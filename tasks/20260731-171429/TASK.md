# Split the Telegram surface under the size cap

- STATUS: OPEN
- PRIORITY: 85
- TAGS: refactor, v0.2.0, telegram, backend, maintainability
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT
- PARENT: 20260731-171411
- DEPENDS ON: 20260731-171420

## Story

As a maintainer, I want the Telegram surface split from one 1448-line module
into transport, command handling, and rendering, so that a bot change does not
load the whole surface.

## Steps

- [ ] Characterize existing Telegram behavior with `tests/test_telegram.py` and
      `tests/test_telegram_approvals.py` before moving code.
- [ ] Separate transport/polling wiring, command and callback handlers, and
      message rendering into distinct modules.
- [ ] Keep approval and host-operation flows in one place; they are a security
      boundary, not a rendering concern.
- [ ] Apply the epic comment policy to every file touched.
- [ ] Remove the corresponding allowlist entries from the size guard.

## Definition of Done

- No Telegram module exceeds 600 lines and no allowlist entry remains
  (cmd: `python scripts/check_file_size.py`).
- Command, callback, approval, and rendering behavior unchanged
  (cmd: `python -m pytest tests/test_telegram.py tests/test_telegram_approvals.py`).
- Full backend gate passes (cmd: `nix flake check`).
- `scufris/README.md` module map matches the new layout
  (cmd: `rg -n "telegram" scufris/README.md`).

## Notes

- Epic: 20260731-171411.
- Depends on: 20260731-171420.
- Telegram shares orchestrator paths that 20260729-103712 will also touch. Do
  not extract a shared orchestrator service here; that seam belongs to 103712.
