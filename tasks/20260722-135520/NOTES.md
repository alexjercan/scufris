# NOTES: opencode serve against host llama-server (de-risk)

Input contract for the backend task (20260722-135525). Everything here was
probed live against `opencode serve` 1.17.9 and the host `llama-server`
(127.0.0.1:11433) on 2026-07-22.

## Provider config (opencode.json)

`examples/opencode/opencode.json` declares a custom OpenAI-compatible provider:

```json
{
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama.cpp (local)",
      "options": { "baseURL": "http://127.0.0.1:11433/v1" },
      "models": { "gemma-4-26B-A4B-it": {}, "gemma-4-12B-it": {}, "Qwen3.6-35B-A3B": {} }
    }
  }
}
```

opencode reads it from `$OPENCODE_CONFIG` (absolute path to the file) or an
`opencode.json` in the cwd. `opencode models` then lists both `llamacpp/<id>`
(the provider ID form) and `llama.cpp/<id>` (the display-name form).

**Provider/model id form the backend must send: `providerID="llamacpp"`,
`modelID="gemma-4-26B-A4B-it"`.** Use the provider ID (`llamacpp`), not the
display name.

## Daemon + health

`opencode serve --port <p> --hostname 127.0.0.1` starts the headless HTTP
daemon. `GET /global/health` -> `{"healthy":true,"version":"1.17.9"}`. When
`OPENCODE_SERVER_PASSWORD` is unset the daemon logs "server is unsecured" and
takes no auth; when set it is the HTTP Basic *password* with an empty username
(`httpx.BasicAuth("", password)`), per the reference client.

## One turn (synchronous send)

- `POST /session` `{"title": "..."}` -> `{"id": "ses_...", ...}`.
- `POST /session/{id}/message` blocks until the model finishes and returns
  `{"info": {...}, "parts": [...]}`. Assistant text = concatenation of every
  `parts[i].type == "text"` part's `.text`. Tool activity shows up as
  additional part types (see below).

Request body schema (from the daemon's own OpenAPI at `GET /doc`):

```
POST /session/{sessionID}/message
  parts     (required)  array of TextPartInput | FilePartInput | AgentPartInput | SubtaskPartInput
  model     {providerID, modelID}         # both required when present
  agent     string                        # select a named agent (permission preset)
  tools     { <toolName>: boolean }        # per-request enable/disable of tools
  system    string
  variant   string
  messageID string (^msg)
  noReply   boolean
  format    OutputFormat
```

TextPartInput = `{"type": "text", "text": "..."}`.

## Permission mechanism (manual|edit|auto mapping)

opencode gates tools two ways, BOTH confirmed available on the message endpoint:

1. **`tools` boolean map (per request) - RECOMMENDED for the backend.** Disable
   the mutating tools to enforce read-only; no shared config state. Tool names
   are opencode built-ins: `bash`, `edit`, `write`, `patch`, `read`, `grep`,
   `glob`, `list`, `webfetch`, `todowrite`, `todoread`, `task`. Proposed map:
   - `manual` (read-only): `{"edit": false, "write": false, "patch": false, "bash": false}`
   - `edit`: `{"bash": false}` (edit/write/patch allowed, no shell)
   - `auto`: `{}` (send nothing - all tools available)
2. **Named agents (config-side) + `agent` field.** Define agents in opencode.json
   with `permission` blocks (`allow`/`ask`/`deny` per tool) and pass `agent` per
   request. NOTE: in a headless server there is no one to answer `"ask"`, so a
   config using `ask` would stall/deny; only `allow`/`deny` are safe headless.
   opencode's own `--auto` flag auto-approves non-denied, but that is a CLI flag,
   not a per-request server control - hence mechanism (1) is preferred.

The backend should implement mechanism (1) (`tools` map) as `_OPENCODE_PERMISSION`.
Verify the exact behaviour live once the model is loaded (does disabling `edit`
actually make the tool unavailable to the model) - deferred to 135525 with the
model warm.

## Tool-calling quality (gemma-4-26B-A4B-it)

Observed live (model warm). Two turns:

- Plain prompt ("Reply with exactly: hello from gemma") -> clean text reply
  `hello from gemma`, `parts` = one `text` part, 0 tool parts. Works.
- Tool-requiring prompt ("Use your tools to list the files in the current
  directory, then tell me how many there are") with tools AVAILABLE (default) ->
  parts were `step-start`, `text`, `step-finish` with **NO tool-call part**. The
  model answered `There are 3 files in the current directory.` - a FABRICATION
  (it never ran `list`/`glob`/`bash`, so it cannot have counted).

Verdict: gemma-4-26B-A4B-it does not reliably invoke opencode's tools; it
answers from priors instead. This confirms the spike's premise (local models are
weak at tool-calling) and the accepted decision to ship the backend anyway - the
gate is a completed turn, not agentic tool fidelity. For the backend (135525):
do not depend on the model driving MCP tools; the read-only `manual` mode is the
safe default and tool reliability is a model-quality follow-up, not a blocker.

## Gotchas

- **Cold load can be VERY SLOW - and it re-downloads on upstream revision
  changes.** The host `llama-cpp` service uses `hf-repo`/`hf-file`, so a model's
  first use resolves the HF repo and downloads the GGUF into the service's
  private cache (`/var/cache/private/llama-cpp`, root-only) before loading. The
  trigger is not only "first EVER use": when `ggml-org` re-uploads the repo, the
  file hash changes and llama.cpp fetches the NEW blob even though an older one
  is cached. Seen live 2026-07-22: `gemma-4-26B-A4B-it` Q8_0 (~26GB) had a Jun-26
  blob on disk but re-downloaded a new `...downloadInProgress` blob (~26GB, ~40
  min) because the upstream revision moved. With `cudaSupport=true` the weights
  load into VRAM, so the worker's system RSS stays low (~80-190MB) THROUGHOUT -
  low RSS does NOT mean "not loading/downloading"; check `GET /v1/models`
  `status.value` (`unloaded`->`loading`->`loaded`) and the blobs dir instead.
  Implication for the backend: the turn timeout must tolerate a multi-minute
  (occasionally tens-of-minutes) first turn, and pre-warming is advisable.
- **Cache hygiene.** After a revision re-download the old blob is orphaned but
  not auto-removed. Reclaim with `huggingface-cli delete-cache --dir
  /var/cache/private/llama-cpp` (only deletes detached revisions). To avoid
  surprise re-downloads, pin a revision in the `models-preset` `hf-file` or set
  `HF_HUB_OFFLINE=1` on the service (offline also blocks legit updates).
- Model ids appear twice in `opencode models` (provider-id vs display-name
  form); always use the provider-id form (`llamacpp/...`) in API calls.
- The turn is synchronous over HTTP; live token streaming needs the separate
  `GET /event` SSE bus (deferred to a backend follow-up).

## Verified turn transcript

Live, 2026-07-22, `opencode serve` (port 14096) -> `llamacpp/gemma-4-26B-A4B-it`
-> host llama-server :11433:

```
$ OPENCODE_URL=http://127.0.0.1:14096 OPENCODE_MODEL=gemma-4-26B-A4B-it \
    python examples/opencode/prove_turn.py "Reply with exactly: hello from gemma"
url:      http://127.0.0.1:14096
model:    llamacpp/gemma-4-26B-A4B-it
session:  ses_07664bb25ffe5tvcsRt9u1zNxl
modelID:  gemma-4-26B-A4B-it
tools:    0 tool part(s)
reply:
hello from gemma
OK
```

Health probe (`examples/opencode/check_health.py`) -> `healthy: True`,
`version: 1.17.9`.
