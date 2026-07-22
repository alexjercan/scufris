# Retro: Stand up opencode serve + prove one turn (gemma-4-26B-A4B-it)

- TASK: 20260722-135520
- BRANCH: spike/opencode-serve-llamacpp (landed 8e04298)
- REVIEW ROUNDS: 1 (APPROVE, out-of-context)

## What went well

- The de-risk-before-backend split paid off: proving the opencode -> llama-server
  pipeline and recording its exact shapes (provider/model id form, message
  schema, permission mechanism) as a NOTES contract means 135525 is now
  mechanical, with the live-verification gotchas already surfaced.
- Out-of-context review re-ran every DoD proof itself (health, live turn, ruff,
  mypy, pytest, `opencode models`) and cross-checked NOTES honesty - a real
  independent pass, not a rubber stamp. It caught a genuine MINOR (exit code not
  gating on the healthy flag).
- Once I recognized the load was an opaque, I/O-bound external wait, moving it
  to a background job and continuing (writing NOTES, probing the permission API)
  was the right call - the turn completed on its own clock and notified.

## What went wrong

- I burned ~15 min of foreground wall-clock polling a network-bound model load
  before backgrounding it, and I stated "it's downloading" as fact when it was
  an inference. Root cause: I inferred external state from a proxy metric - flat
  worker RSS (~80-190MB) - and concluded "not loading / must be downloading".
  With `cudaSupport=true` the weights load into VRAM, which never shows in
  system RSS, so the metric was structurally incapable of answering the
  question. The authoritative sources (llama-server `GET /v1/models`
  `status.value`, and the on-disk blobs dir) were available the whole time; the
  blobs dir is what finally showed a `...downloadInProgress` blob and the truth
  (an upstream-revision re-download, not a fresh one).
- Two diagnostic detours (a du-growth check on the wrong cache, a stuck-vs-slow
  debate) both trace to that same proxy-metric misread.

## What to improve next time

- For any wait on an external service, query that service's own status API
  before inferring from OS-level proxies (RSS, CPU, process age). Never read
  "model loading" off process RSS - CUDA/VRAM makes it meaningless.
- Recognize an opaque external wait within a minute or two and background it
  immediately; do not foreground-poll a network/disk-bound operation.

## Action items

- [x] LESSONS: `query-service-status-not-os-proxy` (new)
- [x] LESSONS: `hf-refetches-on-upstream-revision-change` (new)
- [x] NOTES.md already carries the cache-hygiene + revision-pinning guidance
      for the operator; no separate follow-up task needed.
- No code follow-up: the MINOR was fixed in-cycle; NITs accepted.
