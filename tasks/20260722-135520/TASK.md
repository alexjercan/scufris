# Stand up opencode serve against existing host llama-server (:11433) + prove one turn with gemma-4-26B-A4B-it

- STATUS: CLOSED
- PRIORITY: 20
- TAGS: spike, agent, backend, nix
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED

## Story

Before writing the `OpenCodeBackend` (20260722-135525), de-risk the moving
parts that live OUTSIDE scufris: an `opencode.json` custom provider pointing
`opencode serve` at the existing host `llama-server` (127.0.0.1:11433,
OpenAI-compatible `/v1`), and the exact request/permission API the backend will
drive. The deliverable is a proven, documented config plus a runnable example
that completes one real turn against `gemma-4-26B-A4B-it`, and a NOTES.md that
records the concrete shapes (provider/model id form, per-tool permission API,
tool-calling behaviour) so the backend task is mechanical.

This is a spike-flavoured task: its output is a config file, an example, and a
NOTES.md - not production Python behind the seam.

## Steps

- [x] Write `examples/opencode/opencode.json`: a custom provider (id e.g.
      `llamacpp`) using `@ai-sdk/openai-compatible` with
      `options.baseURL = "http://127.0.0.1:11433/v1"` and the host models
      (`gemma-4-26B-A4B-it`, `gemma-4-12B-it`, `Qwen3.6-35B-A3B`) declared. No
      real API key (llama-server needs none; use a dummy if opencode insists).
- [x] Verify opencode discovers it: from `examples/opencode/`,
      `opencode models` (or `GET /provider`) lists `llamacpp/gemma-4-26B-A4B-it`.
      Record the exact provider/model id string the backend must send.
- [x] Start the daemon: `opencode serve --port 4096` (in the dev shell), and
      confirm `GET /global/health` returns `{healthy: true, version}`.
- [x] Prove one turn: `POST /session` then `POST /session/:id/message` with
      `{model:{providerID:"llamacpp",modelID:"gemma-4-26B-A4B-it"}, parts:[{type:"text",text:"..."}]}`;
      assert a non-empty assistant `text` part comes back.
- [x] Port the reference health probe to `examples/opencode/check_health.py`
      (adapt scufris-bot `examples/check_opencode_health.py`) and add
      `examples/opencode/prove_turn.py` doing the create+send round-trip and
      printing the reply. Both read `OPENCODE_URL` (default
      `http://127.0.0.1:4096`) and `OPENCODE_SERVER_PASSWORD`.
- [x] Probe the per-tool PERMISSION API opencode exposes (the shape that maps to
      manual|edit|auto): inspect how `send_message`'s `tools`/`agent`/`system`
      or session/agent config gates `edit`/`bash`. Record the exact mechanism.
- [x] Write `tasks/20260722-135520/NOTES.md`: provider/model id form, the
      health+turn transcript, the permission mechanism, tool-calling quality
      with gemma (does it call tools or emit JSON), and any gotchas (model load
      latency on first turn, `jinja`, `/v1` path, timeouts). This is the input
      contract for 20260722-135525.

## Definition of Done

- `opencode serve` health probe succeeds against a daemon configured for
  llama-server (cmd: `python examples/opencode/check_health.py` prints
  `healthy:  True` and a version).
- One turn completes end to end against `gemma-4-26B-A4B-it` (cmd:
  `python examples/opencode/prove_turn.py` prints a non-empty reply).
- The provider config is committed and points at :11433 (cmd:
  `grep -n 11433 examples/opencode/opencode.json`).
- NOTES.md records the provider/model id form, the per-tool permission
  mechanism, and the tool-calling observation for gemma-4-26B-A4B-it (cmd:
  `grep -n permission tasks/20260722-135520/NOTES.md`).
- manual: the proven turn's reply is coherent (llama-server is actually
  answering, not erroring or echoing).

## Notes

- Spike: tasks/20260722-135404/SPIKE.md
- Reference infra: scufris-bot @ feature/opencode-v2 (opencode serve + llama.cpp)
  - `scufris_server/opencode_client.py` (client to adapt in the next task),
    `examples/check_opencode_health.py` (health probe to port).
- Test model: gemma-4-26B-A4B-it on host llama-server :11433 (verified reachable
  2026-07-22: `GET /v1/models` lists it; models load on demand).
- opencode 1.17.9 and `sprout` are on PATH; default branch is `master`.
- Blocks: 20260722-135525 (that task consumes this NOTES.md).

## Outcome

Proven end to end: `opencode serve` 1.17.9, configured via
`examples/opencode/opencode.json` (custom `llamacpp` provider ->
`http://127.0.0.1:11433/v1`), drives the host llama-server and returns a real
reply from `gemma-4-26B-A4B-it` (`hello from gemma`). Health probe green. The
provider/model id form (`providerID="llamacpp"`, `modelID="gemma-4-26B-A4B-it"`),
the message-endpoint schema, the permission mechanism (per-request `tools`
boolean map preferred over named agents), and the weak tool-calling observation
are all recorded in NOTES.md as the input contract for 20260722-135525.

### What changed
- `examples/opencode/opencode.json` - the custom OpenAI-compatible provider.
- `examples/opencode/check_health.py`, `prove_turn.py` - standalone probes
  (httpx only, no scufris imports; the reusable client lands with 135525).
- `tasks/20260722-135520/NOTES.md` - the de-risk record + backend contract.

### Difficulties
- The first turn hung ~40 min in `loading`. Diagnosis was muddied by two wrong
  assumptions I had to correct with evidence: (1) flat worker RSS (~80-190MB) is
  NOT "not loading" - with `cudaSupport=true` weights go to VRAM, not RSS; (2)
  it was not a first-ever download but a RE-download: `ggml-org` had re-uploaded
  the repo, so llama.cpp fetched a new ~26GB blob despite a Jun-26 blob being
  cached (confirmed by a `...downloadInProgress` blob in the root-only service
  cache, which needed the operator's sudo to see). Lesson folded into NOTES:
  check `GET /v1/models` status + the blobs dir, not RSS, and budget the turn
  timeout for a tens-of-minutes cold/revision-refetch load.
- gemma-4-26B-A4B-it did not invoke tools on a tool-requiring prompt (fabricated
  a file count) - expected per the spike; recorded, not a blocker.

### Self-reflection
- I burned wall-clock foreground-polling a network-bound load. Better: recognise
  an opaque external wait sooner, background it, and continue - which I
  eventually did. The RSS misread cost two diagnostic detours; the general rule
  (verify the mechanism, do not infer state from a proxy metric) is exactly the
  AGENTS.md lesson and I should have reached for `GET /v1/models` status first.
