# Retro: journal_* tools fail from the operator tool console

- TASK: 20260727-005013
- BRANCH: fix/journal-console-den-env
- REVIEW ROUNDS: 1 (out-of-context; APPROVE, 2 NITs)

See TASK.md for the root cause; process only here.

## What went well

- The operator's report pinned the exact split (works from the agent, fails from the
  RUN button), which pointed straight at the two execution models: agent = MCP
  SUBPROCESS with injected env; console = IN-PROCESS `mcp.call_tool` reading the
  dashboard's own `os.environ`. Reading how the console already bridges the analogous
  `SCUFRIS_API_BASE` (`_ensure_api_base`) gave a ready-made pattern to mirror, so the
  fix was small and consistent instead of inventing a new mechanism.
- Wrote the end-to-end console test so its assertion (`"not configured" not in text`)
  holds in BOTH environments: with `today` present it returns the day, without it
  ("today not found on PATH") the den gate still passed - so the test proves the fix
  in the pure `nix flake check` sandbox too, not just where `today` happens to exist.
- A/B'd the load-bearing test before trusting it (red without the endpoint call).

## What went wrong

- The bug shipped in the original journal-tools task (20260720-122514): the tools were
  designed only for the SUBPROCESS/env-injection model, and the in-process console
  path was never exercised for them. Root cause: the previous task's tests all drove
  the tool functions directly with a monkeypatched env; none went through the real
  `/api/agent/tools/{name}/run` endpoint, so the "console has no injected env" gap was
  invisible. A feature reachable by two execution paths needs a test on each path.

## What to improve next time

- When a tool is exposed through more than one runner (agent subprocess AND the
  in-process operator console), add a test that drives it through the console
  endpoint, not only the tool function - the runners have different env/context and a
  tool can pass one while failing the other.

## Action items

- [x] Fixed: `_ensure_den_path` bridges `settings.den_path` -> `SCUFRIS_DEN_PATH` in the
  console endpoint; console `journal_show` no longer reports "not configured".
- [x] Ledger: added `tool-reachable-by-two-runners-needs-a-test-per-runner`.
