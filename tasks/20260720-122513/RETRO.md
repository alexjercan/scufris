# Retro: persist tool-call chips across session reload

- DATE: 20260720
- VERDICT: shipped

## What went well

- Grounding the fix on the REAL rollout before writing code paid off twice: I
  inspected an actual session to learn the event ordering (user -> commentary ->
  tool -> final_answer -> token_count) and the enum-style result (`{"Ok"}`), so the
  correlation logic matched reality instead of a guess, and the reproduction test
  used the true shape. Then a real-rollout smoke confirmed the fix end to end
  (host_stats/disk_usage/list_processes restored).
- The import-cycle problem had a clean answer already implied by the codebase:
  `agent.py` imports `sessions.py`, and `sessions.py` already owns the other
  rollout models (SessionInfo/Context/UsageQuota), so moving ToolCall/TokenUsage
  there (re-exported from agent) fit the existing seam with zero caller churn.
- Extracting the 3-line reconstruction into an exported `transcriptReply` made the
  new behavior unit-testable without driving the whole `switchSession` fetch flow.

## What went wrong / friction

- Nothing of substance. The only judgement call was per-turn USAGE correlation
  (the token_count that carries a turn's output tokens arrives AFTER the final
  answer), solved with a one-slot `awaiting_usage` buffer. Accepted it as
  best-effort - chips are the reported bug; the token number is a bonus.

## Lesson

- No new ledger entry; this reinforces the existing `harvest-the-stream-you-already-run`
  lesson (the tool-call data was already on disk in the rollout - the fix was
  parsing what exists, not adding a new source) and `probe-runtime-on-target-host-early`
  (read a real rollout to learn the event shape before coding the parser).

## Follow-ups

- Natural extension (noted, not built): clicking a chip could show
  args/result/duration - the rollout's `invocation.arguments` + `result` already
  carry it. Good candidate to fold into a future affordance pass.
- Round-3 remaining (spike 20260720-122301): 122514 (den tools, blocked on the
  unified-CLI project decision), 122515 (slash-commands), 122516 (attachments),
  122517 (settings console), 122518 (projects sub-spike), 122519 (nixos reconcile).
