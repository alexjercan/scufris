# Spike: LLM agent harness for GPT-5.5 via Pro/Plus subscription

- DATE: 20260719-153040
- STATUS: RECOMMENDED
- TAGS: spike, backlog, agent, llm

## Question

Which LLM agent harness should Scufris embed to power the chat agent, given the
hard constraints: drive an external provider's model (target GPT-5.5),
authenticate via a **Pro/Plus subscription rather than a metered API key**, and
integrate cleanly from Python? Evaluate opencode, OpenAI Codex, and comparable
harnesses and recommend one.

## Context

The agent is the third Scufris pillar; per the user's clarification the agent
does the heavy lifting of running tools (e.g. `tatr`), while the dashboard stays
read-only. Scufris is a Python app (FastAPI). The subscription-not-API-key
constraint is the crux and rules out the plain metered API path. The harness
also needs tool/function calling so the agent can run curated local commands
(scoped in [[20260719-153050]]).

Findings below are from web research (July 2026); GPT-5.5 and these harnesses
are past the assistant's training cutoff, so this rests on the cited sources,
not memory. Sources are listed at the end.

## Options considered

- **OpenAI Codex via the official `openai-codex` Python SDK (RECOMMENDED).**
  Codex is OpenAI's coding agent (native Rust CLI). It supports "Sign in with
  ChatGPT" so usage bills against a Plus/Pro/Business plan instead of API
  credits, and there is an **official Python SDK** (`pip install openai-codex`,
  Apache-2.0, published by OpenAI, v0.144.x as of 2026-07-17) that spawns a
  local `codex app-server` and talks JSON-RPC over stdio. API shape:
  `Codex()` / `AsyncCodex()` context manager, `thread_start(model=, sandbox=)`,
  `thread.run(prompt) -> TurnResult`; auth helpers `login_chatgpt()`,
  `login_chatgpt_device_code()` (headless), `login_api_key()`. Models: GPT-5.6
  family (`gpt-5.6-sol/terra/luna`) plus `gpt-5.5`; GPT-5.5 launched
  subscription-only, GPT-5.6 spans ChatGPT/Codex/API. Custom tools are provided
  via **MCP** (register a stdio MCP server under `[mcp_servers.<id>]`), which is
  exactly the seam the tool-execution spike needs. Pros: first-party Python SDK
  (no HTTP-server plumbing), subscription auth meets the hard constraint, GPT-5.x
  available, MCP tool mechanism, sandbox/approval policies. Cons: it is a
  *coding agent* framed as threads/turns, not a chat endpoint (fine for our
  chat-about-the-host use, but the shape is agentic); 0.x, breaking changes -
  pin the version; MCP tool calls are auto-cancelled in fully non-interactive
  `codex exec` unless approval/tool policies are set (the app-server SDK path is
  the right one for us).
- **opencode via `opencode serve` HTTP (runner-up).** MIT, provider-agnostic
  (75+ providers), genuinely headless: `opencode serve` exposes an OpenAPI HTTP
  server (sessions, messages, SSE events); custom tools are TS files or MCP
  servers. But: no official Python SDK (drive it over HTTP or subprocess; it is
  a TS/Bun binary run as an external service), and its **ChatGPT Plus/Pro login
  is the shakiest subscription path** - a native-auth regression (#27905) pushed
  users to a third-party plugin (`opencode-openai-codex-auth`) that itself wraps
  OpenAI's Codex OAuth. So for the specific "OpenAI model via ChatGPT
  subscription" goal, opencode is a less direct, more fragile route to the same
  underlying Codex OAuth. Strong if we later want provider-agnosticism (Claude
  Pro/Max and Copilot subscription logins are more first-class there).
- **Thin custom harness over a subscription transport (rejected).** Maximum
  control, maximum maintenance, and the only subscription transports are either
  Codex's OAuth (so just use Codex) or reverse-engineered ChatGPT-web wrappers
  (`revChatGPT` et al.) which **violate OpenAI's terms**, are anti-bot-fragile,
  and risk account bans. Do not build on scraping.
- **Concede the constraint: use an API key (documented fallback).** The only
  cleanly-within-ToS programmatic path is a metered API key; OpenAI explicitly
  recommends API keys for automation/CI. This contradicts the user's stated
  preference, so it is not the primary, but it is the honest fallback and should
  be a supported auth mode behind the same interface.

## Recommendation

Adopt **OpenAI Codex through the official `openai-codex` Python SDK**, with
**"Sign in with ChatGPT" (device-code) as the primary auth** and an **API key as
a fallback auth mode**, all behind a small Scufris `Agent` interface so the
harness/provider is swappable.

Why it beats the runners-up:

- It is the only option that satisfies all three hard constraints at once:
  OpenAI model + subscription auth + a first-party Python SDK. opencode reaches
  the same OpenAI subscription only through a fragile third-party plugin and has
  no Python SDK; a custom harness means scraping (ToS violation) or reinventing
  Codex.
- Its MCP tool mechanism is the natural home for Scufris's curated tools
  (`tatr`, read-only host info) - see [[20260719-153050]] - so the two agent
  spikes share one design instead of two.
- Keeping the harness behind an `Agent` interface makes the ToS gray area (below)
  and the 0.x churn cheap to reverse: swap to opencode or to API-key auth without
  touching the chat UI or the tool definitions.

### ToS reality (must be recorded, not glossed)

Using a ChatGPT subscription to drive a custom automated app is a **gray area**.
OpenAI's own guidance recommends API keys for automation and confines
subscription-in-automation to trusted, private, single-machine, non-concurrent
use as *your own* account; it has neither blessed nor clearly banned custom-app
use, and declined to confirm it when asked (openai/codex discussion #8338).
Therefore Scufris treats subscription auth as **personal, single-user,
single-machine** use, authenticating as the operator via the official Codex
OAuth device flow (not scraping), with the API-key mode available for anyone who
wants the clean path. This posture is the deciding design constraint, not a
footnote. Rate limits under subscription are a rolling 5-hour window plus weekly
caps, so the agent is fine for interactive chat but not for bursty/high-volume
automation.

## Open questions

- Exact `openai-codex` SDK surface for streaming responses (TS SDK has
  `runStreamed`; the Python doc mentions streaming but did not name the method) -
  confirm at implementation for a responsive chat panel.
- Whether the app should shell out to `codex app-server` (as the SDK does) or use
  `codex exec --json` for simpler turns - the SDK path is the default; revisit
  only if the SDK is too heavy.
- Model default: `gpt-5.5` (the user's stated target) vs a GPT-5.6 tier - pick at
  implementation based on what the operator's plan exposes; keep it a config
  knob.
- Codex is 0.x with breaking changes across minors - pin the version and the SDK,
  and budget for upgrade churn.

## Next steps

Direction-level tasks this spike seeded, for `/plan` to break into steps:

- tatr 20260719-162356: integrate Codex as the agent backend via the
  `openai-codex` Python SDK, behind a Scufris `Agent` interface, with ChatGPT
  device-code auth primary and API-key fallback.
- tatr 20260719-162406: add the agent chat panel to the dashboard (chat with the
  agent from the UI, streaming replies).

The agent's tool-running model (exposing `tatr` and read-only host info to the
agent via MCP) is owned by [[20260719-153050]].

## Fix record

- 20260719-162356 (agent backend): built the `Agent` interface + a Codex backend
  behind it. The `openai-codex` Python SDK proved un-installable in the uv2nix
  venv (bundled binary), so it stayed operator-installed and mock-tested.
- 20260719-164418 (NixOS runtime): the recommendation's *capability* held (Codex
  + ChatGPT subscription = the way to GPT-5.5), but the *Python integration
  mechanism changed*: instead of the SDK, the agent drives the nixpkgs `codex`
  CLI via `codex exec` as a subprocess. LIVE-VERIFIED returning a real GPT-5.5
  reply on the host. The `Agent` interface made the swap internal.

## Sources

- Codex auth / Sign in with ChatGPT: https://learn.chatgpt.com/docs/auth ,
  https://learn.chatgpt.com/docs/auth/ci-cd-auth
- Codex Python SDK: https://pypi.org/project/openai-codex/ ,
  https://github.com/openai/codex/tree/main/sdk/python
- Codex non-interactive mode / app-server / MCP:
  https://learn.chatgpt.com/docs/non-interactive-mode ,
  https://developers.openai.com/codex/app-server ,
  https://learn.chatgpt.com/docs/extend/mcp
- Codex models (GPT-5.5 / 5.6): https://learn.chatgpt.com/docs/models ,
  https://openai.com/index/gpt-5-6/
- ToS gray area: https://github.com/openai/codex/discussions/8338 ,
  https://openai.com/policies/service-terms/
- opencode server / providers / tools: https://opencode.ai/docs/server/ ,
  https://opencode.ai/docs/providers/ , https://opencode.ai/docs/custom-tools/ ,
  https://github.com/sst/opencode
- opencode ChatGPT-auth regression + plugin:
  https://github.com/anomalyco/opencode/issues/27905 ,
  https://github.com/numman-ali/opencode-openai-codex-auth
- Subscription rate limits: https://help.openai.com/en/articles/20001106-codex-rate-card
