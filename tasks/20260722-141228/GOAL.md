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

- [x] 20260722-135520 (p20, scufris) Stand up opencode serve vs host llama-server + prove one turn (gemma-4-26B-A4B-it)
      landed 8e04298; 1 review round (APPROVE, out-of-context); proved a real gemma-4-26B-A4B-it turn (`hello from gemma`); NOTES = backend contract for 135525. Gotcha found: HF revision-refetch cold-load.
- [x] 20260722-135525 (p10, scufris) Add opencode serve backend behind AgentBackend + settings/auth plumbing [depends on 135520]
      landed 861f2c4; 1 review round (APPROVE, out-of-context); OpenCodeBackend verified live end-to-end (real gemma turn + read_status/transcript). Filed 20260722-153555 for the pre-existing red mypy gate.

## Manual acceptance (batched for the user at Finish)

The turn-level manual DoDs (a coherent reply from gemma-4-26B-A4B-it) were
confirmed LIVE during 135520 and 135525 by both the implementer and the
out-of-context reviewer ("hello from gemma" / "backend works"). What remains is
real-app acceptance the user should do when convenient:

- (pending) Exercise the opencode backend through the actual scufris app: start
  `opencode serve` (OPENCODE_CONFIG=examples/opencode/opencode.json), set
  SCUFRIS_OPENCODE_URL, create an `opencode` agent from /agents (or set
  SCUFRIS_AGENT_BACKEND=opencode for the orchestrator), and confirm a chat turn
  streams and the status/transcript panels populate. Expect weak tool-calling
  (a known, accepted trait of the local model - see the spike).

## Done-definition status (Finish verification)

1. opencode serve health + one live gemma-4-26B-A4B-it turn - MET (135520).
2. Backend.OPENCODE + OpenCodeBackend stream/read_status/read_transcript - MET
   (135525, unit + live).
3. Permission modes map + settings/auth plumbing + .env.example - MET.
4. Harness-level test exercising the backend - MET (test_opencode_backend + a
   live get_backend("opencode") run).
5. Overall "nix flake check green" - PARTIAL / DEFERRED: ruff + pytest (312) +
   frontend suite are green and the backend adds ZERO net-new mypy errors, but
   the mypy check is RED on master already (44 pre-existing tests/ type errors,
   no pydantic.mypy plugin). Deferred to tatr 20260722-153555 (green the gate);
   not caused by this goal.
