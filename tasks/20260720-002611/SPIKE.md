# Spike: token-by-token streaming + reasoning + events via codex

- DATE: 20260720-002611
- STATUS: RECOMMENDED
- TAGS: spike, agent

## Question

The user wants the chat to stream the assistant's reply TOKEN BY TOKEN, and to
show "thinking" (reasoning) and tool calls / other events live - "as much as
possible if not all". The current SSE streaming (tatr 20260719-223103) only shows
tool completions + a timer, and the text lands all at once. Is token-by-token +
reasoning even obtainable from codex, and how?

## Context

The agent drives `codex exec --json` as a one-shot subprocess per turn
(`scufris/agent.py`). We stream its stdout line-by-line over SSE. The prior spikes
assumed exec is "turn-level"; this spike VERIFIES that against real turns and
finds the streaming path that actually carries deltas.

## Findings (probed on this host, codex 0.144.4)

- **`codex exec --json` stdout is turn-level. Proven.** A real turn emits only:
  `thread.started`, `turn.started`, `item.completed` (the FULL `agent_message`
  text or an `mcp_tool_call`), `turn.completed` (usage). No deltas. Setting
  `-c show_raw_agent_reasoning=true -c model_reasoning_summary=detailed` changed
  nothing on stdout.
- **The rollout files are also completed-item granularity.** They write the full
  `agent_message` (phase=final_answer), reasoning SUMMARIES (only when the model
  produces them), tool items, and `token_count` - but NO token deltas. A grep of
  ALL rollouts for any `*delta*` event returned nothing. So tailing the rollout
  buys reasoning-summary + richer items, but still not token-by-token.
- **`codex app-server` (experimental) IS the streaming protocol** the TUI/IDE use.
  Its generated JSON Schema (`codex app-server generate-json-schema`) defines
  exactly the delta notifications we need: `outputDeltaNotification` (agent text
  token-by-token), `ReasoningTextDeltaNotification` /
  `ReasoningSummaryTextDeltaNotification` ("thinking" token-by-token),
  `AgentMessageThreadItem` / `ReasoningThreadItem`, `PlanDeltaNotification`,
  `ProcessOutputDeltaNotification`, and thread-item events for tools. It is a
  persistent JSON-RPC-over-stdio server with a documented (but EXPERIMENTAL)
  schema (`generate-ts`/`generate-json-schema` exist), sharing codex auth.

Conclusion: token-by-token + reasoning + full events are ONLY available via
`codex app-server`. `codex exec` fundamentally cannot do it.

## Options considered

- **A. Migrate the agent backend to `codex app-server` (RECOMMENDED, with a
  probe-first plan).** Drive a persistent `codex app-server` process over
  JSON-RPC: initialize, start a thread/turn, and consume its notification stream
  (`outputDelta` -> append text token-by-token; `ReasoningTextDelta` -> a live
  "thinking" section; thread-item / process events -> a live event feed; final
  usage). Wire behind the existing `Agent` seam so it is a second backend
  (config-gated: `agent_backend = exec | app_server`), keeping `codex exec` as the
  proven fallback and for the CLI. Forward the deltas over the existing SSE
  endpoint as new event kinds. Pros: delivers EXACTLY the ask (token text +
  reasoning + all events); it is codex's real streaming interface. Cons: a large
  new client (JSON-RPC handshake, request/response correlation, thread/turn
  lifecycle, sandbox/approval via the protocol); the protocol is EXPERIMENTAL and
  may churn across codex versions; more moving parts than a one-shot subprocess.
  De-risk by PROTOTYPING the handshake + one streamed turn FIRST (its /plan's
  opening step) before committing the full backend.
- **B. Enrich the exec streaming (rejected as insufficient).** Tail the rollout
  during a turn to surface reasoning summaries + tool items, add a "thinking..."
  state and better chips. No token-by-token (impossible on exec). Lower risk but
  does NOT meet the core ask; keep the ideas as fallback polish if A is deferred.
- **C. Do nothing (rejected).** The current one-chunk reply is what the user is
  asking to move beyond.

## Recommendation

Go with A: add a `codex app-server` streaming backend behind the `Agent` seam,
config-gated so `codex exec` stays as the fallback/CLI path, and forward its
`outputDelta` / reasoning-delta / event notifications over SSE so the UI can
render text token-by-token, a live "thinking" section, and an event feed. Plan it
PROBE-FIRST: the first step is a throwaway-grade prototype that completes the
JSON-RPC handshake and captures a real streamed turn's deltas, to confirm the
experimental protocol + auth before building the production client. Two flows:

1. **Backend (tatr 20260720-002619)** - the app-server JSON-RPC client + streamed
   turn, behind the Agent seam, new SSE event kinds (text delta, reasoning delta,
   event). Probe the protocol first.
2. **Frontend (tatr 20260720-002621)** - render token-by-token text into the
   assistant bubble, a collapsible/live "thinking" section for reasoning deltas,
   and a live event feed (tools, plan, process). Builds on the new SSE events.

## Open questions

- **Experimental-protocol risk**: the app-server schema may change across codex
  versions. Mitigation: pin behind a config flag, keep `codex exec` as fallback,
  and generate/inspect the schema at build/plan time. Confirm the user accepts an
  experimental dependency for token-by-token.
- **JSON-RPC handshake specifics** (initialize params, the method to start a
  turn, sandbox/approval wiring, session resume) are unknown until the probe -
  the /plan's first step resolves them from `generate-json-schema` + a live run.
- **Concurrency/lifecycle**: one long-lived app-server process vs one per turn;
  how session switch/fork/delete map onto app-server threads. Resolve at /plan.

## Next steps

- tatr 20260720-002619: Agent backend via codex app-server - stream token/reasoning/tool deltas (probe-first)
- tatr 20260720-002621: Chat UI - token-by-token text, thinking section, live event feed

## Fix record

(Appended by each implementing task as it lands.)
