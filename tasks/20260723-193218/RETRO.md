# Retro: spike - claude scufris MCP parity

- TASK: 20260723-193218
- DATE: 20260723
- OUTCOME: landed, self-review (research + live-probe), seeded impl 20260723-201851

## What we set out to do

Answer "how do we give a claude sub-agent the scufris tools" empirically, before
committing to an implementation - the Codex-first parity gap.

## What went well

- PROBED THE REAL CLI instead of reasoning (the repo's load-bearing lesson
  `probe-runtime-on-target-host-early`). `claude --help` + a live turn nailed the
  exact flags (`--mcp-config` inline JSON, `--strict-mcp-config`, `--allowedTools
  mcp__scufris__<tool>`, `--permission-mode default`) and PROVED claude actually
  calls `mcp__scufris__request_input` unattended. The whole mechanism works - the
  impl is now formatting, not discovery.
- Caught the `--mcp-config` variadic/greedy gotcha ONLY because I ran it: a
  `--mcp-config "$JSON" mcp list` swallowed the subcommand as config paths. That
  would have been a maddening impl bug; now it is a documented constraint + a test.
- The same server binary + role env works verbatim across backends (`apply_role`
  scopes by env, not by codex), so the DECISION (backend-agnostic core, per-backend
  formatters) fell out naturally and mirrors T2's `role_tool_names` anti-drift move.
- Confirmed cheaply without burning the subscription on discovery: `--help` for
  flags, one tiny "reply OK" turn for the auth/parse boundary, one steered turn for
  the tool_use proof. Three turns total.

## What went wrong / friction

- The stream also showed unrelated tool names (`rust-analyzer-lsp`, `ToolSearch`)
  despite `--strict-mcp-config`. Did not chase it (the scufris tool_use was
  unambiguous), but the impl's live DoD should double-check strict isolation so a
  sub-agent's claude turn does not inherit the operator's global MCP servers.

## Lessons (candidates for the ledger)

- `claude-mcp-config-is-variadic-bound-it-with-a-flag`: `claude --mcp-config
  <configs...>` is greedy - it consumes every following token as another config path
  until the next `--flag`. In argv, always follow `--mcp-config <json>` with a flag
  (`--strict-mcp-config` / `--allowedTools` / `-p`), never a positional, or the
  prompt/subcommand gets eaten ("config file not found: .../<token>").
- `claude-mcp-tool-approval-is-allowedTools-not-permission-mode`: to run an MCP tool
  UNATTENDED on claude, `--permission-mode` is not enough - allowlist the tool by its
  `mcp__<server>__<tool>` name via `--allowedTools`; then `--permission-mode default`
  does not hang. (Proven: request_input fired with `--allowedTools
  mcp__scufris__request_input --permission-mode default`.)

## Deferred / seeded

- IMPL 20260723-201851: wire it into the claude backend from a shared core, with
  argv tests + the live loop as DoD. The remaining unknowns (orchestrator allowlist
  wildcard, `--resume` + `--mcp-config`, disabled-tools passthrough, strict
  isolation) are its live-probe DoD.
