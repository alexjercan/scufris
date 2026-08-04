# Assemble provider context from the semantic conversation

- PRIORITY: 98
- TAGS: feature, v0.2.0, lane1, chat
- KIND: TASK
- ACTIVITY: -
- GATES: -
- RESOLUTION: -
- PARENT: 20260801-154211
- DEPENDS ON: 20260804-115256

## Story

As the operator, I want the conversation to survive `/new`, compaction, a
backend switch and a restart, so that the provider session is a cache I can
throw away rather than the place my history actually lives.

Without this the release's headline promise is unimplementable. Nothing in the
sprint plan covered it before the 2026-08-04 lane cut.

## Steps

- [ ] Make the provider session a CACHE keyed
      `(conversation, backend, policy version)`, per
      `tasks/20260729-220835/DECISION.md` section 1. A key miss is normal, not
      an error.
- [ ] Re-seed an invalid session from context assembled out of the semantic
      conversation.
- [ ] Answer SUMMARY VERSIONING here and record it in `DECISION.md`. A summary
      produced under one assembly policy is not interchangeable with one
      produced under the next; the version is what makes a stale summary
      detectable rather than silently wrong.
- [ ] Answer EAGER-VERSUS-LAZY RE-SEED on a backend switch in the same record.
      Eager pays the cost at switch time and is predictable; lazy pays it at the
      next turn and can surprise the operator mid-sentence. Pick one, write down
      the rejected one.
- [ ] Keep assembly BOUNDED. The decision's own Consequences warn that assembly
      "becomes code Scufris owns and must keep bounded"; an unbounded assembler
      turns a long conversation into a provider error at the worst moment.
- [ ] Grow `examples/chat_conversation.py` to switch backend mid-script and
      re-print, with an assertion that the semantic transcript is identical and
      the provider session id is not.

## Definition of Done

- Switching backend re-seeds from the conversation and preserves the semantic
  transcript exactly (test: `test_backend_switch_preserves_the_conversation`).
- Assembled context stays within its configured bound for a conversation far
  larger than the bound (test: `test_assembled_context_is_bounded`).
- A summary written under an older policy version is not reused
  (test: `test_stale_summary_version_forces_reassembly`).
- A valid cached session is reused rather than re-seeded
  (test: `test_valid_session_is_not_reseeded`).

## Notes

- Source: `tasks/20260729-220835/DECISION.md` section 1 and its Consequences.
- Two of the six questions the spike deferred "to v0.3.0 tasks" are answered
  here rather than in a container that no longer exists.
- Lane 1 of `tasks/20260801-154211/TASK.md`.
