# Goal: opencode serve llama.cpp self-hosted agent backend

- DATE: 20260722
- UMBRELLA TASK: 20260722-141228
- LANDING SCOPE: squash-merge each task to master, no push. Standard flow.

## Goal

Add a third agent backend to scufris - `opencode` - that drives a self-hosted
llama.cpp model through `opencode serve`, sitting behind the existing
`AgentBackend` seam next to codex and claude. The point is twofold: a hackable
local backend, and proof the seam is not codex/claude-shaped (a JS-native agent
driven over an HTTP event bus is structurally unlike both). The self-hosted
model already runs on the host (`services.llama-cpp`, 127.0.0.1:11433, models
including `gemma-4-26B-A4B-it`), so scufris does NOT package llama.cpp; it points
`opencode serve` at that OpenAI-compatible endpoint via a custom provider.

Direction and rationale are pinned in the spike:
`tasks/20260722-135404/SPIKE.md` (RECOMMENDED). Infra is ported/adapted from the
proven `scufris-bot` reference (`feature/opencode-v2`). Tool-calling on a local
model is known-weak; shipping anyway is an explicit, accepted decision - the
gate is a completed turn, not flawless agentic tool-use.

## Done means

1. `opencode serve` runs in the Nix devshell pointed at the host llama-server
   (:11433) via a custom OpenAI-compatible provider in `opencode.json`, its
   `/global/health` returns healthy, and one turn completes end to end against
   `gemma-4-26B-A4B-it` (cmd: the seeded example / health probe against the
   running daemon; manual: eyeball the turn's reply).
2. `Backend.OPENCODE` exists and `get_backend("opencode")` returns an
   `OpenCodeBackend` implementing `stream` / `read_status` / `read_transcript`
   behind `AgentBackend`, driving `opencode serve` over an adapted async HTTP
   client (test: pytest covers the backend + client parsing).
3. Permission modes `manual|edit|auto` map to opencode's per-tool permission
   config, and the settings/auth plumbing exists (`OPENCODE_URL`,
   `OPENCODE_SERVER_PASSWORD`, a `local`/`none` auth mode, llama-server base
   URL), documented in `.env.example` (cmd: `grep` the new keys).
4. At least one harness-level test exercises the backend the way the app does
   (test: an integration test that streams a turn, real daemon or a faithful
   fake).

Overall: `nix flake check` (ruff + mypy + pytest) passes on master after both
tasks land.

## Tasks

Updated as tasks land (one line per land, like a spike's Fix record).

- [ ] 20260722-135520 (p?, scufris) Stand up opencode serve vs host llama-server + prove one turn (gemma-4-26B-A4B-it)
- [ ] 20260722-135525 (p?, scufris) Add opencode serve backend behind AgentBackend + settings/auth plumbing

## Manual acceptance (batched for the user at Finish)

Accumulates `manual:` DoD items as tasks land; presented at Finish.

- (none yet)
