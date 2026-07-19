# Retro: agent reach - config-driven MCP registry + more Scufris tools

- DATE: 20260719
- VERDICT: APPROVE (1 review round)

## What went well

- The registry refactor was made safe by keeping it backward-compatible: with an
  empty `mcp_servers` the `-c` output is byte-identical to the old hard-coded
  block, so no existing codex-exec behavior shifted. Verifying that in the live
  smoke (printing the args) was worth more than any unit assertion.
- Extracting `_format_processes` as a pure function let the process-table tool be
  unit-tested deterministically (fixed `ProcessList` -> exact rows), while the
  real `list_processes`/`disk_usage` got light integration tests. The AGENTS.md
  "prefer integration + a small pure core" split fell out naturally.
- Treating a config-supplied server `id` as untrusted (it becomes a TOML key)
  and validating `^[A-Za-z0-9_]+$` + reserving `scufris` closed a real injection
  vector before it existed - the same instinct as the earlier glob-escape fix.

## What went wrong / friction

- `test_tools_registered` asserted the tool set with `==`, so adding tools broke
  it (by design - it is a canary). Updating it to the new exact set + a
  descriptions-non-empty check is the right call; an exact-set assertion catches
  an accidental tool addition/removal, which a subset check would miss.

## Lessons

- (No new ledger entry - this reused `capture-real-cli-output-for-parser-tests`
  (real df/ps output shapes), the injection-safety instinct from
  `escape-client-strings-before-glob` (applied to TOML keys here), and the
  integration-test-with-a-pure-core pattern. A sign those are settling in.)

## Follow-ups

- The agent-page expansion spike (20260719-212152) is now fully delivered - all
  four seeded tasks landed. See its Fix record.
- Deferred, if ever wanted: external MCP servers still need their binaries
  packaged (nixpkgs/npx/uvx) and a per-server security review; the registry now
  makes wiring one a config line. A write-capable tatr tool (`tatr_new`) was left
  out to keep the server read-only.
