# Review: T3 - prune the MCP surface (drop tatr_* tools)

- TASK: 20260722-222729
- BRANCH: refactor/prune-mcp-tatr-tools

## Round 1

- VERDICT: REQUEST_CHANGES
- REVIEWER: out-of-context

The removal is clean, correct and well-tested: the DoD grep over `scufris/` is empty,
`test_tools_registered` uses an exact `==` set of the 10 surviving tools (tatr names
genuinely gone, not a weakened `<=`), the trimmed `STEERING_PREAMBLE` parses and reads
cleanly, re-pointed tests assert on real surviving tools, `test_projects._tatr_new` is
a legitimate local tatr-CLI helper (correctly left), the `needs_tatr` marker is still
used (not orphaned), no unused imports, vitest passes, and the SPIKE Q4 caveat is
carried forward. One stale reference found:

- [x] R1.1 (MINOR) .env.example:89 - the sample `# SCUFRIS_DISABLED_TOOLS=["tatr_new"]`
  still names a removed tool (the counterpart `config.py` comment WAS updated in this
  diff). An operator copying the sample would set a `disabled_tools` entry matching no
  tool. Change to a surviving tool.
  - Response: fixed - changed to `# SCUFRIS_DISABLED_TOOLS=["disk_usage"]`. Root cause:
    the first sweep used `--include=*.py/*.md/...` extension globs, which skipped the
    extensionless `.env.example`; re-swept by PATH (not extension) and confirmed clean.
    Also corrected the TASK.md self-reflection, which had overstated that the one-pass
    grep caught everything.

## Round 2

- VERDICT: APPROVE
- REVIEWER: in-session (trivial diff - a one-line doc string in .env.example plus the
  TASK.md self-reflection correction; no logic change)

Re-verified: the path-based sweep for `tatr_ls|tatr_show|tatr_new|_TATR_SORTS` over the
whole worktree (excluding tasks/) returns only the intended CHANGELOG entry and the
test_projects CLI helper. R1.1 resolved.

Open `manual:` DoD items: none.
