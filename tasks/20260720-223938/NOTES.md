# A2b: claude (Claude Code headless) backend - probed formats + live check

- DATE: 20260720
- TASK: 20260720-223938

## Live probe (green)

claude 2.1.193 is installed and authenticated here. A trivial unattended turn:

```
claude -p 'Reply with exactly the word PONG and nothing else.' \
  --output-format stream-json --verbose
```

exited 0 in ~1.5s, produced the correct `PONG`, and wrote a session transcript.
So headless claude runs unattended, exactly like codex - the interface's second
backend is real, not hypothetical.

## Stream format (`--output-format stream-json --verbose`)

JSONL, one object per line:
- `{"type":"system","subtype":"init","session_id":<uuid>,"cwd",...,"tools":[...]}`
  - the first line; carries the session id.
- `{"type":"assistant","message":{"content":[<block>,...],"usage":{...}},
   "session_id":<uuid>}` - `<block>` is `{"type":"text","text":...}` or
  `{"type":"tool_use","name":...}` (default mode emits WHOLE assistant messages,
  not token deltas; `--include-partial-messages` would add deltas - not used).
- `{"type":"result","subtype":"success"|"error"...,"result":<text>,
   "session_id":<uuid>,"num_turns",...,"usage":{...}}` - the terminal line.

`parse_claude_stream` maps: text block -> `StreamTextDelta`, tool_use ->
`StreamTool`, result success -> `StreamDone(session_id=...)`, result error ->
`StreamError`. Tested against these captured shapes.

## Session transcript format (for read_status)

`<claude_home>/projects/<cwd-hash>/<session_id>.jsonl` where `<cwd-hash>` is the
cwd with `/` replaced by `-`. JSONL with `type` in {`queue-operation`, `user`,
`assistant`, ...}. A `user` turn carries `message.content` as a STRING (a
tool_result turn carries a list, not counted). `assistant` carries
`message.content` blocks + `message.usage`.

Crucially: the session file is found by session-id glob under `projects/`, so
`read_status(settings, session_id)` needs NO cwd - the SAME signature that works
for codex. This is the concrete evidence the A2 interface is not codex-shaped
(spike decision 1's whole point): a genuinely different backend (different CLI,
different output format, different on-disk store) slots behind the identical
`AgentBackend` protocol with zero interface changes.

## Deferred to A3 (not blockers here)

- `--permission-mode` write gating (the per-agent cwd-scoped write opt-in) and
  image attachments - land with the run wiring in A3.
- claude `context_window` in status is left 0 (the per-turn assistant `usage` in
  the session file has no window field; the stream `result.modelUsage` does, a
  cheap future add).
