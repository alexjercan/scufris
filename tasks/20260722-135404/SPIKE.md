# Spike: which agent CLI harness for a llama.cpp self-hosted backend

- DATE: 20260722-135404
- STATUS: RECOMMENDED
- TAGS: spike, agent, backend

## Question

Scufris wants a third agent backend that drives a **self-hosted model served by
llama.cpp** (llama-server, OpenAI-compatible `/v1`), sitting behind the existing
`AgentBackend` seam next to codex and claude. The uncertainty this spike
reduces: **which agent CLI harness do we wrap?** The user pinned the frame:

- Motivation: **experiment / hackability** and **prove the backend seam** is not
  codex/claude-shaped (a genuinely different third harness is the point).
- Capability bar: **full agentic** - edit files and run commands under the
  existing `manual|edit|auto` permission modes, not chat-only.
- Integration: an **external CLI harness** driven as a subprocess (like the
  codex/claude backends), NOT a direct OpenAI client we hand-roll.
- Packaging: the harness **must be in nixpkgs** (available from `nix develop`,
  the same standard as `pkgs.codex` / the `claude` CLI).

A good answer names one harness, shows it clears every bar with evidence, keeps
the losers on record, and leaves the choice reversible behind the seam.

## Context

The `AgentBackend` protocol (`scufris/backends.py:104`) is the one seam the
orchestrator, supervisor, store and dashboard depend on. A backend implements
three methods, and they set the concrete "fits the seam" bar a harness must
clear:

1. `stream(...)` - run one turn in the agent's project `cwd`, resume a
   `session_id` when given, map `permission_mode` to the harness's own
   edit/run flags, and yield normalized `StreamEvent`s (text deltas, tool
   calls, done, error). `ClaudeBackend.stream` (`backends.py:425`) is the model
   to copy: spawn the CLI headless, read stdout line by line, parse each JSON
   event into a `StreamEvent`.
2. `read_status(...)` - a READ-ONLY progress snapshot (turns, tool calls,
   tokens, last message, mtime) derived from the session's **durable on-disk
   log**. So the harness must persist a session we can read back.
3. `read_transcript(...)` - parse that same log into `TranscriptMessage`s to
   rebuild the chat.

Distilled, a candidate harness must have: (a) a scriptable **headless mode with
a machine-readable streaming output**; (b) **session persistence + resume by
id**; (c) a **readable transcript** on disk or via a local API; (d) a
**configurable OpenAI-compatible provider** so it points at llama-server;
(e) **agentic edit+run with permission gating**; (f) **presence in nixpkgs**.

Prior art: spike `tasks/20260719-153040` chose codex over opencode for the
*subscription-auth* constraint (drive GPT-5.5 via a Pro/Plus login, no API
key). That constraint is gone here - a local llama.cpp server needs no
subscription and takes any OpenAI-compatible client - which is exactly why the
harness that lost there can win here.

## Options considered

nixpkgs presence was checked against the repo's pinned `nixpkgs`
(`nix eval nixpkgs#<pkg>.version`): `opencode` 1.17.9, `aider-chat` 0.86.1,
`goose-cli` 1.28.0, `crush` 0.80.0, `open-interpreter` 0.4.2, `codex` 0.142.2
all resolve; `gptme` and `plandex` do NOT (eliminated on bar (f)).

- **opencode** (1.17.9) - RECOMMENDED. `opencode run --format json` runs
  headless and emits raw JSON events on stdout - a direct match for the
  ClaudeBackend stdout-line-parse pattern. `--session <id>` / `--continue` /
  `--fork` give session resume; `--model provider/model` plus a custom
  OpenAI-compatible provider in `opencode.json` (`options.baseURL` ->
  llama-server) point it at the self-hosted model; `--agent`, `--auto` and
  per-tool `permission` config (allow/ask/deny for edit and bash) map cleanly
  onto `manual|edit|auto`. It ALSO ships a headless server (`opencode serve`,
  REST `POST /session/:id/message`, SSE `GET /event`), so it fits the codex
  `app_server` pattern too if we ever want long-lived sessions. Sessions/
  messages are retrievable (`GET /session/:id/message`) for read_status/
  read_transcript, with an on-disk store as the fallback source. Fresh,
  well-documented llama.cpp story (multiple July-2026 walkthroughs). Cons: the
  on-disk store layout is under-documented (mitigated: read back via the local
  server API, or `run --format json` output); another JS runtime CLI to wrap.

- **goose-cli** (1.28.0, Block / now Linux Foundation) - strong runner-up.
  Headless `goose run`, session resume (`--session-id` / named sessions), an
  OpenAI-compatible provider that takes a custom base URL, and a rich
  MCP-extension tool system (very agentic). Loses to opencode on the machine
  interface: its `run` output is oriented to human/markdown rendering, so the
  normalized event stream is harder to parse than opencode's first-class
  `--format json`. Best fallback if opencode's local tool-loop disappoints.

- **crush** (0.80.0, Charmbracelet) - viable. `crush run` non-interactive,
  SQLite-backed sessions per project, `type: openai` + `base_url` custom
  provider with `/v1/models` auto-discovery. Two strikes: reported local-model
  tool-calling flakiness (charmbracelet/crush#2073 - local model emits JSON
  instead of invoking a tool), and a SQLite session store that is readable but
  heavier to parse than line-delimited JSON. Keep as a third option.

- **aider-chat** (0.86.1) - weak seam fit. Mature and OpenAI-compatible, but
  interactive-first: no clean JSON event protocol for streaming, edits are
  git-diff driven, and command execution is not gated by an edit/run permission
  model that maps to `manual|edit|auto`. More rewrite-the-adapter than
  wrap-the-CLI.

- **open-interpreter** (0.4.2) - eliminated. Older, REPL-shaped, weaker
  session/permission/streaming story than the leaders; no advantage here.

- **codex pointed at llama.cpp** (0.142.2, already a backend) - explicitly
  considered and rejected FOR THIS GOAL. codex supports custom `model_providers`
  (base_url) and OSS models, so aiming the existing `CodexBackend` at
  llama-server would cost almost no code and reuse app_server + rollout reading
  + resume wholesale. But it adds no new harness: it neither proves the seam is
  codex-agnostic nor satisfies the "external CLI harness (e.g. opencode)" and
  "hackability" intent the user pinned. Worth remembering as the cheapest path
  if the goal were ever just "any local backend"; it is not what this spike was
  asked to find.

## Recommendation

Wrap **opencode** as the third backend (`Backend.OPENCODE`). Post-spike the
user pinned two decisions that sharpen the shape:

- **Drive `opencode serve`, not `opencode run`.** Use the long-lived HTTP
  daemon fronted by a typed async HTTP client (`POST /session`,
  `POST /session/:id/message`, `GET /event` SSE bus, `GET /global/health`),
  auth via `OPENCODE_SERVER_PASSWORD` as HTTP Basic password. This is the
  **codex `app_server` mould**, not the claude subprocess-and-parse one -
  a server you drive and stream a normalized event bus from. `opencode run
  --format json` remains a valid fallback if the daemon proves fiddly, but
  serve is the target.
- **Reuse the proven infra from the `scufris-bot` reference**
  (github.com/alexjercan/scufris-bot, branch `feature/opencode-v2`), where
  `opencode serve` against llama.cpp already works. Port/adapt its
  `scufris_server/opencode_client.py` (`OpencodeClient`: httpx.AsyncClient,
  pydantic response models with `extra="allow"`, error taxonomy
  network/client/server/unavailable/stale-session, `/event` reconnect
  sentinel) and its `examples/check_opencode_health.py` health probe rather
  than deriving the client from scratch. Its `nix/modules` show the service
  wiring.

opencode clears every bar with the least adapter code: an event bus we can
normalize into `StreamEvent`s, sessions created/resumed/read back over HTTP
(`GET /session/:id/message` powers read_status/read_transcript), per-tool
permission config mapping onto `manual|edit|auto`, a configurable
OpenAI-compatible provider, and it is in the pinned nixpkgs (1.17.9). It also
stresses the seam in the intended way - a JS/TUI-native agent driven over an
HTTP event bus, structurally unlike both codex (JSON-RPC app_server) and claude
(stream-json), so making it fit is the real test that `AgentBackend` isn't
accidentally shaped around the first two.

**The model already exists on the host.** The user runs `llama-server` on the
NixOS box (`services.llama-cpp`, port **11433**, ctx 128k, models including
`gemma-4-26B-A4B-it`, `Qwen3.6-35B-A3B`, `gemma-4-12B-it`). So scufris does NOT
package or launch llama.cpp; it points `opencode serve` at that existing
OpenAI-compatible endpoint (`http://<host>:11433/v1`) via a custom provider in
`opencode.json`. The de-risk task shrinks to standing up `opencode serve`
against the running server and proving a turn. **Test subject:
`gemma-4-26B-A4B-it`.**

Keep the decision **reversible behind one knob**: the harness lives entirely
inside a new backend class selected by `Backend` enum value, so swapping to
goose (the runner-up) is a new class + enum member, not a change above the seam.
The genuinely uncertain parameter is not the harness but the **model**: local
models are materially weaker at tool-calling (crush#2073 is direct evidence),
and the user has already SEEN opencode+llama.cpp tool-calling underperform in
the `scufris-bot` reference. The explicit call here is to ship it anyway - "for
the fun of it" / to prove the seam - so a weak tool-loop is an accepted
outcome, not a blocker. The acceptance gate is therefore **one completed turn
via `opencode serve` against `gemma-4-26B-A4B-it`**, not flawless agentic
tool-use. Treat a better tool-loop as a later model swap (or a fall back to
goose); the seam makes both cheap.

## Open questions

- **Read-back source for read_status/read_transcript**: `GET /session/:id/
  message` off the running daemon (the reference client already models these)
  vs. parsing opencode's on-disk store. The HTTP API is the default, matching
  the serve-driven shape; the on-disk store is a fallback if the daemon isn't
  running at status-read time.
- **Auth mode plumbing**: llama.cpp needs no real auth, but the agent store and
  `.env` carry an `auth_mode` per backend (chatgpt/claude_ai/api_key). opencode
  wants a new `none`/`local` auth mode (or reuse `api_key` with a dummy key) -
  a small enum/settings follow-up, flagged so it is not discovered late. Also
  add `OPENCODE_URL` / `OPENCODE_SERVER_PASSWORD` settings (per the reference
  `Settings`) and a llama-server base-URL knob (default `http://<host>:11433/v1`).
- **Permission enforcement fidelity**: confirm live that opencode's `manual`
  mapping is genuinely read-only (the same caveat noted for claude's headless
  `default` mode at `backends.py:435`).
- **`/event` bus is per-server, not per-session**: opencode's SSE bus fans out
  ALL sessions; the backend must filter events to the turn's `session_id` and
  handle the reference's reconnect sentinel (treat a reconnect as a turn-level
  error). Settled in the reference; carry the same policy.

RESOLVED (was open pre-user-input): whether the local model drives the tool
loop well - accepted as "ship it anyway", gate is one completed turn with
`gemma-4-26B-A4B-it` (see Recommendation).

## Reference

- `scufris-bot` @ `feature/opencode-v2`
  (https://github.com/alexjercan/scufris-bot/tree/feature/opencode-v2) -
  a working `opencode serve` + llama.cpp integration to port infra from:
  - `scufris_server/opencode_client.py` - the `OpencodeClient` to adapt.
  - `examples/check_opencode_health.py` - health-probe example to mirror.
  - `nix/modules/scufris.nix`, `nix/hm-modules/scufris.nix` - service wiring.
- Host: `services.llama-cpp` on NixOS, port 11433, ctx 128k; models
  `Qwen3.6-35B-A3B`, `gemma-4-26B-A4B-it` (test subject), `gemma-4-12B-it`.

## Next steps

Direction-level tasks this spike seeded, for `/plan` to break into steps:

- tatr 20260722-135520: stand up `opencode serve` (in the Nix devshell) pointed
  at the EXISTING host `llama-server` (:11433) via a custom OpenAI-compatible
  provider in `opencode.json`, verify `/global/health`, and prove ONE turn end
  to end against `gemma-4-26B-A4B-it`. Port the reference health-probe example.
  Acceptance gate = one completed turn (tool-calling quality not gated).
- tatr 20260722-135525: add the `opencode` backend behind `AgentBackend`
  (`Backend.OPENCODE` + `OpenCodeBackend` driving `opencode serve` over an
  adapted `OpencodeClient`: stream via `/event` filtered to the session,
  read_status/read_transcript via `GET /session/:id/message`, permission-mode
  mapping onto opencode's per-tool config), plus the settings/auth-mode
  plumbing (`OPENCODE_URL`, `OPENCODE_SERVER_PASSWORD`, a `local` auth mode).

## Fix record

(Appended by the implementing tasks as they land.)
