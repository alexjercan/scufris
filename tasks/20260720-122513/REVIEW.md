# Review: persist tool-call chips across session reload

- VERDICT: APPROVE
- ROUND: 1

## Summary

Tool-call chips (and the token count) rendered only on a live turn; reopening a
session dropped them because `read_transcript` returned `{role,text,ts}` and
`switchSession` built no `reply`. Fixed by harvesting the rollout's
`mcp_tool_call_end` events in `read_transcript`, attaching them (and the turn's
output tokens) to the final-answer message, carrying them through
`TranscriptMessage`, and rebuilding the reply on reload via a new
`transcriptReply`. Reproduced first with a failing test; verified on real data
(an actual session restores `[host_stats, disk_usage, list_processes]`, 497 tok).
132 pytest + 85 frontend green, ruff/mypy clean.

## What is good

- Bug playbook followed: a failing `read_transcript` test (correct event ordering
  - user / commentary / tool / final / token_count) reproduced the mechanism
  before any fix, and is now the regression pin. Plus a real-rollout smoke.
- The correlation matches how codex actually records a turn (confirmed by
  inspecting a live rollout): tools sit between the skipped `commentary` and the
  kept `final_answer`, so accumulating per turn (reset on `user_message`) and
  attaching to the final answer is exactly right - not a guess.
- The `ToolCall`/`TokenUsage` move to `sessions.py` is the clean fix for the import
  cycle (agent already imports sessions), and they are re-exported so no caller
  breaks; `sessions.py` is already the home of the other rollout models.
- `transcriptReply` is a tiny pure exported helper, unit-tested directly and via a
  render test, rather than only reachable through the whole `switchSession` fetch
  flow.

## Findings

- MINOR (accepted) - USAGE is best-effort: it comes from the `token_count` event
  immediately after the final answer, buffered via `awaiting_usage`. If a session
  has no post-answer `token_count`, the reloaded message shows the chips but no
  "N tok" (live always shows it). The chips - the actual reported bug - are solid;
  the token number is a bonus. Documented in the task.
- MINOR (accepted) - if a single turn somehow emits two `final_answer` messages
  with no `token_count` between, the tools attach to the first and usage to the
  last. One final answer per turn is the norm (verified), so this is theoretical.
- NOTE - the natural extension (click a chip -> args/result/duration) is left for
  later; the rollout carries `invocation.arguments` + `result`, so the data is
  already there when we want it.

## Verdict

APPROVE. Correct against the real rollout format, reproduced-then-fixed with a
pinning test and a real-data smoke, and the type move is clean. The two findings
are accepted best-effort/theoretical edges, not defects.
