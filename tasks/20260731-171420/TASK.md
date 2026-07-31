# Establish the file-size guard and sweep comment bloat

- STATUS: OPEN
- PRIORITY: 95
- TAGS: chore, v0.2.0, maintainability, kiss
- KIND: TASK
- FLOW STEP: PLANNING
- PLAN STATUS: DRAFT
- PARENT: 20260731-171411

## Story

As a maintainer, I want an enforced file-size cap and one comment policy
applied across the codebase, so that context cost stops growing silently and
later splits have a gate that fails when they regress.

## Steps

- [ ] Add `scripts/check_file_size.py`: fail when a `scufris/` or `web/src/`
      source file exceeds 600 lines or a test file exceeds 900 lines, with an
      explicit allowlist and a failure message naming each offender.
- [ ] Seed the allowlist with every current offender, and record that each
      later child removes its own entries.
- [ ] Wire the guard into the canonical backend gate (`nix flake check`).
- [ ] Sweep `scufris/` and `web/src/` for comments citing task, spike, or
      decision IDs. Delete the lore. Keep the invariant it was wrapping, as a
      statement about the code.
- [ ] Compact real deferred work into `TODO:`/`FIXME:`/`BUG:`/`NOTE:`
      one-liners. Delete comments that restate the code.
- [ ] Keep module, class, and function docstrings; trim them to purpose,
      contract, and non-obvious behavior. Do not delete a docstring to satisfy
      the cap.
- [ ] Record the cap and the comment policy in `AGENTS.md`.

## Definition of Done

- The guard fails on an oversized file and passes on the current tree
  (test: `test_check_file_size_flags_oversized_file`).
- The guard runs in the canonical backend gate (cmd: `nix flake check`).
- No comment cites a task/spike/decision ID
  (cmd: `rg -n "2026[0-9]{4}-[0-9]{6}" scufris web/src`).
- Deferred work uses the four allowed markers only
  (cmd: `rg -n "XXX|HACK" scufris web/src`).
- Behavior unchanged (cmd: `python -m pytest && cd web && npm run ci`).
- `AGENTS.md` states the cap and the comment policy
  (cmd: `rg -n "600" AGENTS.md`).

## Notes

- Epic: 20260731-171411.
- Known ID-citing comments: `scufris/opencode_client.py`, `mcp_host_tools.py`,
  `agent.py`, `backends.py`, `auth.py`, `agent_store.py`, `sessions.py`,
  `mcp_server.py`, `telegram.py`, `enums.py`, `config.py`, and others.
- Allowlist is a ratchet: entries may only be removed, never added.
- Do not combine with any split; this task changes comments and adds a gate.
