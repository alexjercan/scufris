# A2b: claude (Claude Code headless) runner behind the AgentBackend interface

- STATUS: CLOSED
- PRIORITY: 25
- TAGS: spike,agents,backend

## Goal

A second `AgentBackend` implementation: Claude Code in headless mode
(`claude -p --output-format stream-json`), behind the same interface as codex.
Normalize to the contract: session resume (`claude --resume`, its own on-disk
session store vs codex thread-id), MCP config (`--mcp-config` vs codex `-c`),
permission/sandbox model (`--permission-mode` / `--allowedTools` vs codex
`--sandbox`), and the status source (stream-json / session jsonl -> the same
`agent_status` shape). Building the second backend is what proves the A2
interface is not accidentally codex-shaped (decision 1). Includes its own
unattended probe.

## Steps

Format PROBED first (lesson `probe-runtime-on-target-host-early`, just promoted):
`claude -p <prompt> --output-format stream-json --verbose` emits JSONL:
`{"type":"system","subtype":"init","session_id",...}`, then
`{"type":"assistant","message":{"content":[{"type":"text"|"tool_use",...}],"usage":{...}}}`,
then `{"type":"result","subtype":"success"|...,"result":<text>,"session_id",...}`.
Session file: `~/.claude/projects/<cwd-hash>/<session_id>.jsonl` (findable by
session_id glob, NO cwd needed - so `read_status(session_id)` fits both backends
and the interface is proven not-codex-shaped). Live probe already green (PONG).

- [x] Config (`scufris/config.py`): `claude_bin: str | None = None`,
      `claude_home: Path | None = None` (default `~/.claude`).
- [x] `scufris/backends.py`: a pure `parse_claude_stream(lines) ->
      Iterator[StreamEvent]` mapping stream-json lines to events (assistant text
      block -> StreamTextDelta, tool_use -> StreamTool, result success ->
      StreamDone w/ session_id, result error -> StreamError). A
      `_find_claude_session(claude_home, session_id)` (rglob the projects dir).
- [x] `ClaudeBackend` (`name="claude"`): `stream` spawns
      `claude -p <prompt> --output-format stream-json --verbose [--resume <sid>]`
      with `cwd=`, reading stdout line-by-line through `parse_claude_stream`;
      `read_status` parses the session jsonl (turns = user msgs, tools = tool_use
      blocks, last assistant text, last usage tokens, updated_at = file mtime).
      image_paths + write/permission-mode gating are noted as A3 follow-ups.
- [x] `get_backend("claude") -> ClaudeBackend()`; add `"claude"` to
      `agent_store.KNOWN_BACKENDS` so an agent can select it.
- [x] Tests `tests/test_backends.py`: `parse_claude_stream` over the REAL probe
      lines (captured) yields text + done w/ session_id; `ClaudeBackend.stream`
      via a monkeypatched subprocess emitting those lines; `read_status` over a
      fixture session jsonl; `get_backend("claude")` resolves (flip the A2 test
      that asserted it raised); protocol conformance.
- [x] NOTES.md: record the stream-json + session-file formats and the live probe.
- [x] Full check suite green; close-out.

## Definition of Done

- `parse_claude_stream` maps real stream-json to StreamEvents incl. a final
  StreamDone carrying the session id (test: `parse_claude_stream_from_probe`).
- `ClaudeBackend` satisfies `AgentBackend`; `get_backend("claude")` resolves
  (test: `get_backend_resolves_claude`).
- `ClaudeBackend.read_status` returns a normalized snapshot from a claude session
  file found by id (test: `claude_backend_read_status_from_session`).
- Full suite passes (cmd: `nix develop --command bash -c "ruff check . && mypy .
  && pytest -q"`).

## Notes

- Spike: tasks/20260720-221748/SPIKE.md (decision 1 - codex first, claude right
  after).
- Depends on: 20260720-221935 (A2, landed 4d6850a - interface exists).
- The interface signature `read_status(settings, session_id)` needs NO change for
  claude: claude sessions are found by id via a projects-dir glob, mirroring
  codex's `_find_rollout`. This is the concrete evidence the A2 interface is not
  codex-shaped (decision 1's whole point).
- write/permission-mode (`--permission-mode`) + image attach are deferred to A3
  (the gated-write + run wiring); A2b proves the stream + status halves.

## Close-out

What changed:
- `scufris/backends.py`: `parse_claude_stream` (pure stream-json -> StreamEvent
  mapper), `_find_claude_session`/`_iter_jsonl`/`resolve_claude_home`, and
  `ClaudeBackend` (stream shells out to `claude -p ... --output-format
  stream-json --verbose [--resume]` with cwd; read_status parses the session
  jsonl found by id). `get_backend("claude")` resolves it.
- `scufris/config.py`: `claude_bin`, `claude_home`.
- `scufris/agent_store.py`: `"claude"` added to `KNOWN_BACKENDS` (an agent can
  now select it).
- Tests: parse over the REAL captured probe lines (text + done w/ session id) +
  tool_use/error mapping; stream via a monkeypatched subprocess (cwd + --resume +
  stream-json forwarded); read_status over a fixture session file; flipped the A2
  factory test (claude resolves, "nope" raises); protocol conformance.
- NOTES.md: the stream-json + session-file formats and the live probe.

The point of A2b (spike decision 1): a genuinely different backend - different
CLI, different output format (whole assistant messages vs codex rollout), a
different on-disk session store - slots behind the IDENTICAL `AgentBackend`
protocol with ZERO interface changes. `read_status(settings, session_id)` needed
no cwd because claude sessions are found by id-glob, mirroring codex's
`_find_rollout`. The interface is proven not codex-shaped.

Deferred to A3: `--permission-mode` write gating + image attach (with the run
wiring); claude status `context_window` (0 for now).

Result: 239 tests pass (+4), ruff + mypy clean; live claude probe green.

Self-reflection: probing the real stream-json and session-file formats BEFORE
writing the parser (the lesson I promoted in A2) paid off immediately - the
parser was written against captured reality, and the session-by-id-glob insight
(which kept the interface stable) came straight from inspecting the real files,
not guessing.
