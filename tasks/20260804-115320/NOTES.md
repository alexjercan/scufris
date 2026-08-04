# Understanding: context assembly

## What changes

The provider session stops being where the conversation lives and becomes a
cache that can be thrown away. Scufris assembles what the provider needs from
its own event log.

This is the task that makes the release's headline promise implementable. It was
not in the sprint plan before the 2026-08-04 lane cut - it was found missing
while writing the lanes.

## Surfaces

- `packages/chat/src/scufris_chat/` - assembly and the session cache.
- `examples/chat_conversation.py` - grows the backend-switch section.
- `DECISION.md` here - two of the spike's six deferred questions are answered.

## Data and interfaces

The provider session is keyed `(conversation, backend, policy version)`
(`tasks/20260729-220835/DECISION.md` section 1). A key miss is NORMAL - it is a
cache - and must not surface as an error anywhere.

`policy version` in the key is what makes a stale summary detectable. Without
it, a summary produced under an older assembly policy is silently reused and is
silently wrong.

## Sketches

```
   semantic conversation (source of truth, survives everything)
                 |
                 |  assemble(), BOUNDED
                 v
   +-------------------------------+
   | provider session CACHE        |   key = (conversation, backend, policy)
   |   claude  -> sess_abc         |
   |   codex   -> rollout_xyz      |
   +-------------------------------+

   switch backend:   key misses -> re-seed from assembly -> new provider id
   /new:             cache dropped -> conversation untouched
   restart:          cache cold    -> conversation untouched
```

## Shape

Two deferred questions are answered here rather than in a v0.3.0 container that
the re-cut dissolved:

- **Eager vs lazy re-seed on a backend switch.** Eager pays at switch time and
  is predictable. Lazy pays at the next turn and can surprise the operator
  mid-sentence. Both are defensible; the point of the task is to pick one and
  record the rejection.
- **Summary versioning - SETTLED, and settled by not building it.** See below.

Assembly must be BOUNDED. The decision's own Consequences warn that assembly
"becomes code Scufris owns and must keep bounded", and an unbounded assembler
turns a long conversation into a provider error at the worst possible moment.

### Summarization is cut from v0.2.0 (2026-08-04)

The bound is a WINDOW over recent events. There is no summarizer in this
release.

Evidence, in the order it decided the question:

- **The accepted decision never asks for one.**
  `tasks/20260729-220835/DECISION.md` section 1 says an invalid binding is
  "re-seeded from assembled context" - that is the entire requirement. Summaries
  appear only in `SPIKE.md`'s five-item assembly sketch, and the same spike files
  summary versioning under "Not addressed here". It was a sketch detail, never
  promoted.
- **The bound already exists and is not a summarizer.** The spike calls assembly
  "`format_fork_seed` generalized". That function is in the tree today and bounds
  by windowing: `kept = context[-max_turns:]`
  (`scufris/sessions/transcript.py`). Summarization would not generalize it, it
  would replace it.
- **A summarizer conflicts with how this lane is proven.** The Lane 1
  deliverable is an OFFLINE example, no network, gated by
  `tests/test_examples.py`. A Scufris-side summarizer calls a model. Either the
  demo stops being offline, or the summarizer is faked and the demo no longer
  proves the thing it claims. The window has no such conflict.
- **KISS.** The window is a slice expression. A summarizer needs a compactor,
  storage, an invalidation rule, staleness handling mid-turn, and an answer to
  the question the spike itself left open - who writes it, Scufris or the
  provider's own compaction read back.

What does NOT change: the cache key stays
`(conversation, backend, policy version)`. Policy version is not about
summaries - assembly items 1 and 5 are the system/project policy and the
presets legal right now, and both change independently of any summary.

The accepted cost: the provider stops seeing the early part of a long
conversation. The semantic log is intact, nothing is deleted, and the operator
still reads all of it in the UI, so the release promise holds as written. But
"the model forgot what we discussed an hour ago" is a real experience and a
summary is what fixes it.

**Trigger to reopen:** the first time a conversation's window drops context the
operator actually needed. Record it in `DECISION.md` as a deferral with that
trigger, not as a gap.

## Consequences and open questions

- **No longer the largest task in the lane, and it does not split.** Cutting
  summarization leaves assembly and the cache - two things, not three - and puts
  this at roughly the size of the tables task. Before that cut it was the split
  candidate.
- **Interaction with `20260804-115321`:** assembled context must preserve the
  actor distinction. A quotation that becomes `role="user"` on its way to the
  provider reintroduces the exact defect that task exists to fix, one layer
  down. That test lives in 115321 but the code it tests is here, which is why
  115321 depends on this task as well as on 115256.
- **Closed:** summarization is cut, the bound is a window. See above. Summary
  versioning becomes a recorded deferral rather than implemented machinery, so
  this task now carries one open question (eager vs lazy re-seed) instead of
  two.
