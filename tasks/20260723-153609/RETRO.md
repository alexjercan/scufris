# Retro: SC1 sub-agent request_input-when-blocked steering

- TASK: 20260723-153609
- DATE: 20260723
- OUTCOME: landed, 1 review round (APPROVE)

## What we set out to do

Teach a tool-having codex sub-agent to call `request_input` when blocked, by
riding the instruction on its turn prompt - the missing steering half of the
BC1-BC4 comms channel. Pure prompt-borne steering, no new mechanism.

## What went well

- The spike (`tasks/20260723-153339`) had already decided the design (two
  preambles, same sentinels, gated to the tool-having sub-agent), so work was
  a direct translation with no design churn.
- The `agent_id` gate for the new preamble lines up 1:1 with the existing
  `elif agent_id` gate in `_mcp_overrides` that grants the `request_input` tool,
  so "steered" and "actually has the tool" are the same condition by
  construction - the reviewer verified this mirror explicitly and it is the
  crux of the change's correctness.
- Restructured `_steer` from a single `not is_orchestrator` early-return into an
  explicit three-way role branch, which reads cleaner than bolting a second
  condition onto the old guard and made the exclusivity obvious.
- Out-of-context review passed round 1 with only a non-blocking nit; test
  coverage (agent-gets-preamble, orchestrator-ignores-agent_id,
  tools-disabled-with-agent_id, toolless-claude, strip assertion) was complete
  on the first pass.

## What went wrong / friction

- The check tooling (ruff/mypy/pytest) is only on PATH inside the Nix dev shell;
  the first `ruff`/`mypy` invocations failed with "command not found" until I
  wrapped everything in `nix develop -c bash -c '...'`. Cost two dead commands.
- `mypy .` is RED on master with 58 pre-existing errors (test files passing
  plain strings where `Backend`/`AuthMode`/`AgentState` Literals are expected),
  and `nix flake check`'s mypy check inherits the same red. The task DoD asks for
  "mypy green", which is not literally achievable from this baseline. Confirmed my
  four changed files add zero new errors (an inherited red, not mine) and scoped
  the fix out. Filed as its own follow-up task rather than ballooning this diff
  into unrelated test files.
- `nix develop -c ... python -m pytest -q` swallows the final `N passed in Xs`
  summary line through the pipe (only the progress dots survive to the tail), so
  I had to fall back to checking `$?` explicitly to confirm green. Minor, but it
  wasted a couple of round-trips trying to grep a summary that never reached the
  file.

## Lessons (candidates for the ledger)

- `run-repo-checks-inside-nix-develop`: ruff/mypy/pytest live only in the flake
  dev shell; invoke them as `nix develop -c bash -c '...'`, not bare, or the
  first command dies "command not found".
- `nix-develop-pytest-pipe-eats-the-summary`: piping `nix develop -c ... pytest`
  output drops the final `N passed` line; confirm green via the exit code
  (`>/dev/null 2>&1; echo $?`), not by grepping the tail.
- `scufris-mypy-baseline-is-red`: `mypy .` (and the flake mypy check) is red on
  master with a large pre-existing Literal-vs-str baseline in test files; a task
  DoD asking for "mypy green" means "adds no NEW mypy errors" - verify your
  changed files are clean rather than chasing the whole tree green.

## Follow-ups filed

- New tatr task: clean the pre-existing 58-error mypy baseline (test files using
  plain strings for `Backend`/`AuthMode`/`AgentState` Literals) so the flake
  mypy gate is honestly green again.

## Deferred to Finish

- Manual live-probe DoD: run a real codex sub-agent on a task requiring an
  approval it lacks, confirm it calls `request_input` rather than guessing or
  stopping silently. Batched to the flow Finish checkpoint.
