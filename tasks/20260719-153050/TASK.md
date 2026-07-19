# Spike: Agent tool-execution model (agent runs CLI tools like tatr)

- STATUS: CLOSED
- PRIORITY: 0
- TAGS: spike, backlog, agent, tools

## Question

How should the **agent** run local CLI tools (e.g. custom `tatr` commands) on
the host? The agent does the heavy lifting of tool-running - the dashboard does
NOT get "RUN" buttons. So: how are tools declared and exposed to the agent as
callable functions, how is execution run safely (subprocess, timeouts, output
capture), and how do a tool's results flow back into the chat?

## Context

Per the user's clarification (2026-07-19): the dashboard is read-only display
(host stats, and read-only host info such as the list of `tatr` tasks); it does
not run tools on a button press. Tool execution happens through chatting with
the agent. So this is the AGENT's tool-execution model, not a UI action panel.
The interface must line up with whatever the agent harness spike
([[20260719-153040]]) picks for tool/function calling. Running local commands is
the obvious risk surface, so the design is a curated, declared allowlist - never
a free-form shell.

## What a good answer looks like

A recommended pattern for declaring an agent tool (name, typed args, the command
it maps to, how stdout/stderr/exit status surface back to the chat), the safe
execution path (subprocess with argument lists - never a shell string - plus
timeouts and output bounds), and an explicit safety stance (allowlist only, no
arbitrary shell, validate every argument). Concrete enough for `/plan` to expand
into steps. This is distinct from the dashboard's read-only host-info panels,
which are plain GET endpoints (covered by the dashboard/metrics spikes).

## Candidate directions to explore (diverge before converging)

- **Declarative tool registry** (pydantic-described tools) surfaced to the agent
  harness as function/tool definitions from one source of truth.
- **Thin per-tool Python wrappers** - a function per tool (e.g. `tatr_ls`,
  `tatr_new`); more code, tightest control over each command.
- **Harness-native tool mechanism** - if the chosen harness (opencode/Codex/...)
  already has a tool/plugin system, register tools its way instead of inventing
  one.
- **Generic "run this command" tool** with an allowlist gate - simplest, widest
  risk surface; probably rejected but worth naming.

## Notes

- Output per the /spike skill: write `tasks/<id>/SPIKE.md`, seed direction-level
  tasks, close this spike task.
- Security is the headline: never build an arbitrary-command execution path; use
  subprocess with an argument list (no `shell=True`); validate every argument;
  curate the tool list.
- Depends on the agent-harness spike ([[20260719-153040]]) for the tool-calling
  interface. Read-only host-info surfaces (like showing tatr tasks) belong to the
  dashboard, not here.
