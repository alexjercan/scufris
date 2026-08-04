# Agent: steer the model to prefer the scufris MCP tools over raw shell

- PRIORITY: 40
- TAGS: bug, agent, codex
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE

## Goal

When asked host/tatr questions (e.g. "tell me details about this host"), the agent
runs raw shell commands (`uname`/`df`) instead of the curated scufris MCP tools
(`host_stats`, `disk_usage`, `list_processes`, `tatr_*`). Steer codex to PREFER the
scufris tools so those questions produce a tool call (and a tool chip in the UI).

Root cause: `agent.py:_exec_args` passes only the user prompt with no steering
instructions, and codex also has a read-only shell it reaches for by default.

## Notes

- Spike: tasks/20260720-102348/SPIKE.md.
- User feedback: asked "tell me details about this host" and it used bash instead
  of `host_stats`.
- Probe on the target host which codex lever actually works (base/experimental
  instructions via `-c`, an instructions file, or a prepended prompt preamble)
  WITHOUT dropping the read-only sandbox (see lessons `probe-runtime-on-target-host-early`,
  `codex-resume-rejects-sandbox`). A prompt preamble is the low-risk fallback.
- Verify LIVE that a "tell me about this host" turn emits a `host_stats` tool call.
  Applies to both `exec` and `app_server` backends.

## Investigation (live probes on the host)

Reproduced and bisected the lever with live `codex exec` runs (0.142.2, gpt-5.5),
asking "tell me details about this host" with the scufris MCP server registered:

| Lever | shell cmds | MCP tool calls |
| --- | --- | --- |
| Baseline (as shipped) | 16 | 0 |
| Strengthened tool descriptions | 16 | 0 |
| `-c experimental_instructions_file=...` | 12 | 0 |
| `AGENTS.md` via `-C <dir>` | 9 | 0 |
| **Prompt preamble (in the turn text)** | **0** | **3** (host_stats, disk_usage, list_processes) |

Finding: codex has a strong shell bias and IGNORES the soft channels (tool
descriptions, instructions file, AGENTS.md). Only steering carried on the TURN
PROMPT itself flips it. So the fix rides on the prompt.

## Implementation

- `sessions.py`: `STEERING_PREAMBLE` - a sentinel-wrapped (`[scufris-tools]...
  [/scufris-tools]`) instruction block telling codex to prefer host_stats/
  disk_usage/list_processes/tatr_* over shell for host/task questions - plus
  `strip_steering(text)` (its inverse). sessions.py owns both so they cannot drift.
- `agent.py`: `_steer(settings, prompt)` prepends the preamble when
  `agent_tools_enabled` (no-op otherwise). Wired into `_exec_args` (the exec/stream
  backends) and the app_server `turn/start` input - so BOTH backends steer.
- `sessions.py`: `_read_head` (session title) and `read_transcript` (history
  re-render) run `strip_steering`, so the injected block is invisible to the user -
  titles and message bubbles show only what they typed.
- `mcp_server.py`: strengthened `host_stats`/`disk_usage`/`list_processes`
  descriptions to also say "prefer over shell" (reinforcement; not sufficient alone).

## Verification

- End-to-end through the REAL code path (default `app_server` backend, shipped
  `STEERING_PREAMBLE`, via `_stream_app_server`): tools used =
  [host_stats, disk_usage, list_processes], zero shell, no error.
- Unit tests: `_steer` prepends/omits by the tools flag; `_exec_args` carries the
  preamble as the final prompt arg; `strip_steering` round-trips; the session title
  and transcript hide the preamble; the tool descriptions carry the steering text.
  129 pytest + 73 frontend green.
