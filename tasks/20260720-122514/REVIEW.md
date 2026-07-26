# Review: the-den journal MCP tools

- TASK: 20260720-122514
- BRANCH: feature/journal-mcp-tools

## Round 1

- VERDICT: REQUEST_CHANGES
- REVIEWER: out-of-context (verified in-session: R1.1 reproduced, isolation + CLI-contract claims re-derived)

Verification (not findings): the out-of-context reviewer ran all four DoD proofs
green in the worktree, plus `nix flake check` green in the pure sandbox (the
real-CLI tests skip there via `skipif(shutil.which('today') is None)`). It
independently confirmed: global flags `--den`/`-N` precede the subcommand
(against `today --help`); `SCUFRIS_DEN_PATH` is injected ONLY in the
`is_orchestrator` branch so a sub-agent can never reach the journal; `_run` uses
`shell=False` with an explicit argv (a `--tomorrow`-looking task text is stored
literally, no injection); the gating short-circuits before shelling out and a bad
index surfaces the CLI's one-line stderr, never a traceback. In-session I
reproduced R1.1 (`Settings(den_path=Path('~/personal/the-den'))` -> `isdir` False
raw, True expanded) and confirmed the repo convention is `.expanduser()` at use
time (`projects.py:147,182`, `app.py:974,989-992`, `sesh.py:65,102`).

- [x] R1.1 (MAJOR) scufris/mcp_server.py:_journal - a `~` in `SCUFRIS_DEN_PATH`
  is never expanded, so the exact value `.env.example` recommends
  (`~/personal/the-den`) silently disables the tools: `_journal` does
  `os.path.isdir(den)` and passes `den` to `_run` verbatim, and `agent.py`
  injects `str(settings.den_path)` unexpanded; pydantic's `Path` does not expand
  `~`. Diverges from the repo's `.expanduser()`-at-use-time convention. Fix:
  expand in `_journal` (`den = os.path.expanduser(den)`) before the isdir check
  and the `--den` arg, and reflect the resolved path in the error message.
  - Response: Fixed. `_journal` now `os.path.expanduser(den)`s immediately after
    the empty-check, so the isdir check, the `--den` arg and the "does not exist"
    error all use the resolved path. Added `test_journal_expands_tilde_in_den`
    (argv layer, no `today` needed) pinning that a `~`-prefixed `SCUFRIS_DEN_PATH`
    reaches `_run` expanded. Matches `projects.py`/`app.py` convention.

- [ ] R1.2 (NIT) scufris/mcp_server.py:journal_show - `journal_show(offset=0)`
  always emits `-N 0` even for today, unlike the CLI's own default of omitting
  the flag. Harmless (the CLI accepts `-N 0`).
  - Response: Left as-is by design - the explicit `-N <offset>` keeps the argv
    uniform across offsets and the argv-contract test pins it; the reviewer noted
    leaving it is fine.

- [ ] R1.3 (NIT) scufris/mcp_server.py:journal_notes - `journal_notes` has no
  `offset` param, so notes read today-only while `journal_show` exposes other
  days. Matches the TASK spec (notes are today's), so not a defect.
  - Response: No change - matches spec. The docstring already says "today's
    the-den notes", which is accurate.

## Round 2

- VERDICT: APPROVE
- REVIEWER: out-of-context (round-1 reviewer resumed against the new diff;
  in-session A/B-confirmed the new test fails with the expanduser line removed)

R1.1 verified RESOLVED three ways by the out-of-context reviewer: (1) live
end-to-end - with `HOME` pointed at a scratch dir and
`SCUFRIS_DEN_PATH=~/personal/the-den`, `journal_show()` now drives the real
`today` CLI and returns dated JSON instead of the old "does not exist" error;
(2) load-bearing test - `test_journal_expands_tilde_in_den` passes, and FAILS
with the `expanduser` line deleted (also A/B-confirmed in-session: `assert []`);
(3) matches the repo's `.expanduser()`-at-use-time convention. No new findings;
the diff is one `expanduser` line, a docstring note, and one test. Gates green:
`ruff check . && mypy . && python -m pytest -q` -> 524 passed; `nix flake check`
-> all checks passed. R1.2/R1.3 remain NITs left by design. No open manual: DoD
items (all DoD proofs are `cmd:`). APPROVE.
