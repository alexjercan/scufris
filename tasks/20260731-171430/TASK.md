# Split the host, hostd, and auth modules under the size cap

- STATUS: OPEN
- PRIORITY: 80
- TAGS: refactor, v0.2.0, host, security, backend, maintainability
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT
- PARENT: 20260731-171411
- DEPENDS ON: 20260731-171420

## Story

As a maintainer, I want the host, hostd, and auth modules under the size cap,
so that privileged-path changes are reviewable without loading the entire host
stack.

## Steps

- [ ] Characterize behavior with the existing host, hostd, and auth suites
      before moving code.
- [ ] Split `scufris/hostd/actions.py` (774) by verb family, keeping the
      protocol and dispatch surface in one module.
- [ ] Split `scufris/hostconfig.py` (664) by parse/render versus apply.
- [ ] Split `scufris/mcp_host_tools.py` (630) by host domain (stats, network,
      thermal, packages, generations).
- [ ] Trim or split `scufris/auth.py` (608). The deny-by-default middleware and
      the public-path list stay in one module; only genuinely separable pieces
      move.
- [ ] Apply the epic comment policy to every file touched.
- [ ] Remove the corresponding allowlist entries from the size guard.

## Definition of Done

- No file under `scufris/host*/`, `scufris/hostd/`, or `scufris/auth.py`
  exceeds 600 lines (cmd: `python scripts/check_file_size.py`).
- One deny-by-default middleware remains, public paths stay only in
  `scufris/auth.py`, and no route declares its own auth dependency
  (cmd: `rg -n "Depends" scufris/`).
- Host verbs, previews, approvals, audit, and inspection unchanged
  (cmd: `python -m pytest tests/test_host_actions.py tests/test_host_action_api.py tests/test_host_inspection.py tests/test_auth.py`).
- Privileged VM tests pass (cmd: `nix build .#scufris-hostd-vm-test`).
- `scufris/hostd/README.md` and `scufris/host/README.md` match the new layout
  (cmd: `rg -n "actions|hostconfig" scufris/hostd/README.md scufris/host/README.md`).

## Notes

- Epic: 20260731-171411.
- Depends on: 20260731-171420.
- Privileged path. Any doubt about a split resolves toward leaving the security
  boundary intact, even if a file stays near the cap. Record the exception in
  the allowlist with a reason rather than weakening the boundary.
