# Review: steer the agent to prefer the scufris MCP tools

- VERDICT: APPROVE
- ROUND: 1

## Summary

codex answered host questions with 16 raw shell commands and 0 MCP tool calls.
Live probing showed codex ignores the soft channels (tool descriptions,
`experimental_instructions_file`, `AGENTS.md` via `-C`) and only obeys steering
carried on the turn prompt. The fix prepends a sentinel-wrapped `STEERING_PREAMBLE`
to each turn (both backends) and strips it from titles/transcripts so the user
never sees it. End-to-end through the real `app_server` path: tools used =
[host_stats, disk_usage, list_processes], zero shell, no error. 129 pytest + 73
frontend green.

## What is good

- The lever was found by measurement, not guesswork: five variants probed live on
  the host, tabulated in TASK.md. This is the `probe-runtime-on-target-host-early`
  discipline paying off - three plausible "instruction" mechanisms were all duds.
- The steering rides the one channel that works but stays invisible: `sessions.py`
  owns both `STEERING_PREAMBLE` and its inverse `strip_steering`, so the format and
  its stripping cannot drift, and `_read_head` + `read_transcript` both strip it.
- Because `read_transcript` strips, the fork seed (built from it) is clean, and
  `_steer` adds exactly one preamble - no double-injection, no leak into pasted
  context. A nice emergent property of stripping at the read boundary.
- Both backends covered (`_exec_args` and app_server `turn/start`), gated on
  `agent_tools_enabled` (no steering when there are no tools to prefer).
- Verified at two levels: unit tests pin the deterministic wiring (prepend/omit,
  round-trip, title/transcript hiding), and a real-code e2e probe pins the actual
  behavior. The LLM behavior is not faked in a unit test - it is proven by probe.

## Findings

- MINOR (accepted) - the preamble (~90 tokens) is prepended every turn, so it adds
  to `input_tokens` and the context-fill %. Cheap and worth the reliability;
  re-steering each turn is more robust than first-turn-only (which probe C was).
- MINOR (accepted) - the preamble slightly nudges every turn, including non-host
  ones. Its wording ("prefer tools FOR host/task questions; fall back to the shell
  otherwise") scopes it, so general/coding answers are unaffected.
- NOTE - strengthened tool descriptions were kept as reinforcement even though the
  probe proved they are not sufficient alone; they cost nothing and pin a clear
  contract. Honest about their limited role in TASK.md and the test comment.

## Verdict

APPROVE. The root cause was measured, the fix is on the only channel that works,
it is invisible to the user by construction, both backends are covered, and it is
proven end-to-end on the real host. Findings are cost/scope trade-offs, not defects.
