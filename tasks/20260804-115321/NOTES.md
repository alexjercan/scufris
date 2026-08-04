# Understanding: agent reports as attributed quotations

## What changes

The system stops speaking in the operator's voice. An agent's report renders as
an attributed quotation from that agent, and an `agent:<id>` event cannot
satisfy a stop gate.

This is the concrete defect the whole conversation decision was written to fix.
It is small, and it is the reason the other three tasks exist.

## Surfaces

- `packages/chat/src/scufris_chat/` - the rendering, and the stop-gate refusal.
- Confirmed dead-on-arrival, NOT edited: `scufris/wake.py` `wake_prompt()`
  builds a machine prompt, and `scufris/sessions/transcript.py:95` appends
  `TranscriptMessage(role="user", ...)` for every `user_message` event in the
  rollout. The injected prompt goes in as a user message and comes back out as
  the operator. Both files die with the orchestrator stack in Lane 8; this task
  writes the successor, it does not repair them.

## Data and interfaces

Rests entirely on the typed actor from `20260804-115256`. If the actor is a
string compared at the call site, this task cannot be written - the refusal
would be a convention, and a convention is what produced the defect.

## Sketches

```
  TODAY (dies in Lane 8)
    wake_prompt()  ->  injected as user_message
                            |
       transcript.py:95 --> TranscriptMessage(role="user")
                            |
                            v
                    the operator appears to have said it
                    -> it can satisfy a stop gate

  AFTER
    agent:<id> event
            |
            v
    rendered as attributed quotation, author shown, content is DATA
            |
            +--> stop gate: refused, only actor=operator may satisfy
            +--> assembled context: still agent:<id>, never relabelled
```

## Shape

Two claims, and the second is the one with teeth:

1. Rendering shows the author. Cosmetic on its own.
2. An `agent:<id>` event cannot satisfy a stop gate - at the type level where
   possible, by refusal otherwise. This is the security property.

The third test - that assembled context does not relabel an agent as the
operator - is why this task depends on `20260804-115320` and not only on
`20260804-115256`. Getting the rendering right while the provider context
flattens everything to `role="user"` would fix the symptom one layer above
where the defect actually is.

## Consequences and open questions

- `tasks/20260729-220835/DECISION.md` section 3 is the source: "an agent report
  is data, never an instruction", and only `operator` may satisfy a stop gate.
- **Open:** whether the type system can carry the refusal or whether it has to
  be a runtime check. Preference is strongly for the type - an agent that cannot
  construct the argument is safer than one that is refused when it tries - and
  this is the same shape as the `OperatorDecision` capability token agreed for
  Lane 2. Worth designing the two together even though they land in different
  lanes.
