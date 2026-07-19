# Spike: Agent tool-execution model (agent runs CLI tools like tatr)

- DATE: 20260719-153050
- STATUS: RECOMMENDED
- TAGS: spike, backlog, agent, tools

## Question

How should the agent run local CLI tools (e.g. custom `tatr` commands) on the
host? How are tools declared and exposed to the agent as callable functions, how
is execution run safely, and how do results flow back into the chat? The
dashboard does NOT get "RUN" buttons - tool-running is the agent's job.

## Context

Depends on the harness spike [[20260719-153040]], which recommended **OpenAI
Codex via the `openai-codex` Python SDK**. Codex's extension mechanism for
custom tools is **MCP** (Model Context Protocol): the agent is given tools by
registering MCP servers under `[mcp_servers.<id>]` (stdio or streamable HTTP).
That decides the shape of this spike - we do not invent a tool protocol, we
implement an MCP server. The dashboard's read-only host-info surfaces (host
stats, listing `tatr` tasks for display) are plain GET endpoints owned by the
dashboard, not this spike.

## Options considered

- **A Scufris stdio MCP server exposing curated tools (RECOMMENDED).** A small
  Python MCP server (its own process, launched by Codex via
  `[mcp_servers.scufris]`) that declares a fixed set of tools - e.g. `tatr_ls`,
  `tatr_show`, `tatr_new`, `host_stats` - each backed by a typed Python handler
  that runs the underlying command with `subprocess` using an **argument list
  (never a shell string)**. Pros: one source of truth for tools; harness-native
  (Codex just consumes it); the allowlist is literally the set of handlers we
  wrote, so there is no arbitrary-command path; reuses the metrics collector for
  `host_stats`; testable in isolation (call a handler with args, assert the
  result) without the LLM in the loop. Cons: an MCP server to build and keep in
  sync with the harness; MCP tool calls are auto-cancelled in fully
  non-interactive `codex exec`, so the agent must run via the app-server/SDK path
  with approval/tool policies configured (already the harness spike's chosen
  path).
- **Let the agent call `tatr` through Codex's built-in `shell` tool
  (rejected as the primary).** No MCP server to build; gate with
  `approval_policy`/`sandbox_mode`. But it hands the agent a general shell rather
  than a curated verb set - the exact arbitrary-execution surface we want to
  avoid - and gives no typed, testable tool boundary. Useful only as an escape
  hatch behind explicit approval, not the design.
- **Thin per-tool wrappers not exposed over MCP (rejected).** Python functions
  the agent cannot actually call, because Codex's tool intake is MCP. Would force
  a second bespoke bridge. The MCP server already *is* the set of per-tool
  wrappers, exposed the way the harness consumes them.
- **Generic "run this command" MCP tool with an allowlist (rejected).** One tool
  taking an arbitrary argv filtered by an allowlist. Simpler but the widest risk
  surface and the easiest to get wrong; a declared verb per capability is safer
  and self-documenting.

## Recommendation

Build a **single Scufris stdio MCP server** that declares a curated set of typed
tools, registered with Codex under `[mcp_servers.scufris]`. Each tool maps to one
Python handler that:

- validates its typed arguments (pydantic), then runs the concrete command via
  `subprocess.run([...], ...)` with an **argument list, `shell=False`**, a
  timeout, and captured/bounded stdout+stderr;
- returns a structured result (stdout, exit status, parsed payload where useful,
  e.g. `tatr_ls` returning task rows) that Codex surfaces back into the chat.

Safety stance (non-negotiable, from AGENTS.md security posture): allowlist only -
the tools are the handlers we ship, nothing more; never a shell string; every
argument validated/escaped; timeouts and output caps; start with **read-mostly**
tools (`tatr_ls`, `tatr_show`, `host_stats`) and add mutating ones (`tatr_new`,
`tatr_edit`) deliberately, gated by Codex's approval policy so a write is
confirmed. `host_stats` reuses the existing `Collector` from the metrics work
(tatr 20260719-154420) - no second source of host data.

This beats the runners-up because it is harness-native (Codex consumes MCP
directly), gives a typed and independently testable tool boundary, and makes the
allowlist structural rather than a filter. It also keeps the door open: the same
MCP server works for any MCP-capable harness, so if the agent backend is later
swapped (opencode also speaks MCP) the tools move with it.

## Open questions

- Which MCP Python library to use (the official `mcp` SDK vs `fastmcp`) - decide
  at implementation; both give a stdio server with typed tools.
- The initial tool set and which are read-only vs mutating (and thus require
  Codex approval). Start read-only; expand with intent.
- Exactly how `host_stats` and the read-only dashboard host-info endpoints share
  code with the metrics collector without duplicating shaping logic.
- Whether the MCP server runs in-process-adjacent (spawned by Codex) or as a
  Scufris-managed subprocess - follows from the harness integration.

## Next steps

Direction-level task this spike seeded, for `/plan` to break into steps:

- tatr 20260719-162419: build the Scufris MCP server exposing curated,
  allowlisted tools (`tatr_*`, `host_stats`) to the agent, backed by safe
  `subprocess` handlers, registered with Codex under `[mcp_servers.scufris]`.

Depends on the agent backend (tatr 20260719-162356, from [[20260719-153040]])
for how Codex is launched and configured.

## Fix record

(Appended by each implementing task as it lands.)

## Sources

- Codex MCP tool mechanism: https://learn.chatgpt.com/docs/extend/mcp ,
  https://learn.chatgpt.com/docs/config-file/config-reference
- Codex app-server / non-interactive tool-approval behavior:
  https://developers.openai.com/codex/app-server ,
  https://learn.chatgpt.com/docs/non-interactive-mode
- (Harness choice and its sources: tasks/20260719-153040/SPIKE.md)
