# Review: Fix the examples the package carve broke

- TASK: 20260804-041340
- BRANCH: fix/examples-carve-paths

## Round 1

- REVIEWER: out-of-context
- VERDICT: APPROVE

- [ ] R1.1 (MINOR) tasks/20260804-041340/NOTES.md:15 - the "After" paragraph
  still states the pre-DECISION scope ("all thirteen exit 0, and the four join
  the opt-in list"), and the illustrative `OFFLINE` block below it marks
  `comms_loop.py`, `telegram_approval.py` and `telegram_bot.py` as `# + new`.
  DECISION.md narrowed the task to one entry and the branch lands one. Add a
  one-line pointer to DECISION.md under that paragraph, or strike it. Not
  diff-introduced - NOTES.md was committed at PLANNING (`3ae0aa7`) and this
  branch does not touch it - but the branch is what makes it wrong in fact.
  - Response:

Non-findings, recorded as prose:

- The `tests/test_examples.py:7` docstring says "the twelve scripts there"
  while `examples/` holds thirteen. Pre-existing drift, outside these Steps,
  and untouched by the diff.

### What the recording pass re-derived

Both the out-of-context reviewer and this pass ran all five proofs on the
branch independently.

| proof | kind | result |
|-|-|-|
| `uv run python examples/host_agent.py` | cmd | exit 0 |
| `uv run pytest tests/test_examples.py -k host_agent` | cmd | exit 0 |
| `telegram_approval.py` event-loop grep | cmd | grep exit 0 |
| example sweep | manual | exactly `comms_loop.py`, `telegram_approval.py`, `telegram_bot.py` |
| `uv run pytest -q` | cmd | exit 0 |

Load-bearing claims re-derived rather than accepted:

- `test_host_actions.py` really does live at
  `packages/hostd/tests/test_host_actions.py` and nowhere else, so the new
  `sys.path` target in both examples is the one directory that resolves the
  import.
- The `OFFLINE` tuple is strict ASCII sort, and `host_agent.py` sits in its
  correct slot between `core_unit_of_work.py` and `host_report_fixture.py`
  (`_` 0x5F sorts before `c` 0x63, so `host_agent.py` precedes both
  `host_report_fixture.py` and `hostctl_approval_flow.py`).
- `grep -rn 'ROOT / "tests"' examples/` exits 1: no example carries a stale
  carve path, which is Step 3's reconciliation.

The out-of-context reviewer additionally ran
`nix build .#checks.x86_64-linux.pytest --no-link` to exit 0, covering the one
way the tuple addition could pass locally and fail in the sandboxed gate where
`scufris_host` / `scufris_hostd` resolve differently.

The DoD text was not edited on this branch - only checkboxes, `ACTIVITY` and
the Close-out - so no proof was weakened mid-flight. Scope matches DECISION.md
exactly: no package export of `host_files` / `host_runner`, no `flake.nix`
change, and no event-loop fix folded in from `20260803-014210`.

The `manual:` sweep is discharged, not pending: both passes ran it and got the
predicted three-name failure set.
