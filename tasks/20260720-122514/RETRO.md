# Retro: the-den journal MCP tools

- TASK: 20260720-122514
- BRANCH: feature/journal-mcp-tools
- REVIEW ROUNDS: 2 (R1: REQUEST_CHANGES, 1 MAJOR + 2 NITs; R2: APPROVE)

See TASK.md for what/why and the verified `today` CLI contract; this is process only.

## What went well

- Verified the external CLI contract empirically BEFORE writing the wrapper: ran the
  real `today` against a temp den to pin that global flags (`--den`, `-N`) must
  precede the subcommand, that `show` omits notes, and the exact error/exit shapes.
  The `test_journal_argv_contract` test then encodes those facts, so the wrapper was
  built against reality, not a guessed contract.
- Caught the nix-sandbox-vs-external-binary mismatch during WORK, not at verify time:
  `today` is on PATH in `nix develop` only because it leaks from the user profile, and
  is absent in the `nix flake check` sandbox. Designed a two-layer test strategy
  (deterministic argv/gating tests that stub `_run`, always green; real-CLI tests
  guarded by `skipif(shutil.which('today'))`) so the source-of-truth gate stays green
  without coupling the flake to an unpublished local repo.
- The out-of-context round-1 reviewer earned its keep: it found the `~`-expansion
  MAJOR the implementing session was blind to.

## What went wrong

- R1.1 (MAJOR): `den_path` was gated with `os.path.isdir` and passed to `today`
  verbatim, with no `.expanduser()` - so the exact value written into `.env.example`
  (`SCUFRIS_DEN_PATH=~/personal/the-den`) silently disabled every journal tool.
  Root cause: I treated the `new-config-field-updates-all-its-surfaces` lesson as a
  doc-COMPLETENESS rule (add the knob to `.env.example`) and stopped there, instead of
  a CORRECTNESS rule (the documented example must actually work). The repo's own
  `.expanduser()`-at-use-time convention (`app.py`, `projects.py`, `sesh.py`) was
  right there and I did not mirror it. The blind spot was structural: my own tests all
  used absolute `tmp_path` dens, so none exercised the `~` form I had just documented.

## What to improve next time

- When adding a filesystem-path config knob, mirror the repo's expanduser-at-use-time
  convention AND treat the value in `.env.example` as a TEST INPUT: write one test
  that feeds that exact documented form (here `~/...`) through the real code path.
  A doc example you never execute is an untested claim.

## Action items

- [x] Fixed R1.1: `_journal` now `os.path.expanduser`s the den; pinned by
  `test_journal_expands_tilde_in_den` (A/B-confirmed it fails without the fix).
- [x] Ledger: added `expanduser-path-config-at-use-time` and
  `external-cli-tests-skipif-not-flake-coupling`.
- [x] tatr 20260726-225845 (follow-up): put `today` + `SCUFRIS_DEN_PATH` on the
  DEPLOYED scufris service PATH (nixos / home-manager module), like codex/claude/git,
  so the journal tools work off a dev box.
