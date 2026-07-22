# Retro: Add the opencode backend (opencode serve -> llama.cpp)

- TASK: 20260722-135525
- BRANCH: feature/opencode-backend (landed 861f2c4)
- REVIEW ROUNDS: 1 (APPROVE, out-of-context)

## What went well

- The 135520 de-risk paid off exactly as intended: the NOTES contract
  (provider/model id form, message schema, permission `tools` map) made the
  backend mechanical - no live-API guesswork during implementation, and the seam
  fit cleanly (the AgentBackend protocol was genuinely not codex/claude-shaped).
- Probing the real `GET /session/:id/message` shape from the warm daemon before
  writing the parser (per `probe-real-shape` ledger habit) caught the nested
  `info.tokens.{input,output,cache.read}` layout the reference paraphrase hid.
- A live end-to-end run through `get_backend("opencode")` (not just mocks) proved
  the whole stack - stream + read_status + read_transcript - against real gemma,
  and the out-of-context reviewer reproduced it independently. One-vote review
  was enough because the reviewer re-ran everything including the live turn.

## What went wrong

- The `mypy .` baseline confusion cost real time. I saw 45 errors on the branch,
  assumed I'd regressed, and had to baseline master (44) to prove the debt was
  pre-existing. Root cause: I ran `mypy scufris/ tests/` (explicit paths) which
  surfaced tests/ errors the light `mypy scufris/` never shows, then discovered
  the repo's actual gate (`nix flake check` -> `mypy .`) is red on master. Two
  separate facts (the gate command, the pre-existing red) had to be established
  before I could trust "zero net-new".
- Minor: my first read_status design used `asyncio.run` inside a sync method,
  which would have thrown inside the app's event loop. Caught it before running
  by reasoning about the call site; switched to a blocking `httpx.get`. The
  reviewer later confirmed the call sites are sync threadpool handlers, so the
  fix was right but my initial docstring rationale was slightly off.

## What to improve next time

- Establish the repo's REAL gate command and its baseline state at task START
  (run `nix flake check` or read the flake `checks`), not at verify time. Had I
  known `mypy .` was the gate and already red, the "did I regress" detour
  vanishes.
- For a sync protocol method that needs I/O, check the call site's sync/async
  nature first (threadpool vs event loop) before choosing asyncio.run vs a
  blocking client - the answer drives the design.

## Action items

- [x] LESSONS: `establish-the-real-gate-and-its-baseline` (new)
- [x] tatr 20260722-153555 filed: green the pre-existing red mypy gate (pydantic
      plugin + enum-typed test args). Not this branch's scope.
- [x] Follow-ups already recorded in NOTES: live `/event` token streaming and
      image attachments are deferred; both fit behind the same seam.
