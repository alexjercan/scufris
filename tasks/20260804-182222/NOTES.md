# Understanding: where OperatorDecision lives

## What changes

One import path is decided, and the boundary tests are made to express the
decision rather than tolerate it. Small task, but it gates two later ones.

## Surfaces

- `packages/chat/src/scufris_chat/decisions.py` - where the type is today.
- `packages/chat/src/scufris_chat/README.md` section 3.1 - books the move.
- `tests/test_package_boundaries.py:227` `DECLARED_GRAPH`, checked for EQUALITY.
- `tests/test_package_boundaries.py:49` `CORE_MODULES` - an allowlist, and its
  docstring says adding an entry "is meant to cost a line here and a
  justification in the task record".

## Data and interfaces

`authorize(conn, conversation_id, event_seq) -> OperatorDecision` reads the
`event` table. It cannot move out of `chat` under any option; only the TYPE is
in question.

The type carries `conversation_id`, `event_seq`, `actor: Actor` and a
module-private witness. `Actor` is `chat`'s, which is what makes option A harder
than the epic assumed.

## Sketches

```
  A. move the TYPE to core            B. add the edge hostctl -> chat
     core: OperatorDecision, Actor?      hostctl imports scufris_chat
     chat: authorize (the mint)          type and mint stay together
     hostctl -> core   (already)         graph gains one edge
     flow    -> core   (already)
     COST: core stops being domain-      COST: the privileged host client
     free, or Actor stays in chat        depends on the conversation package
     and the type loses its actor

  C. approve() leaves hostctl entirely
     hostctl: propose / apply / deny / audit only
     the decision-taking caller sits above both
     COST: "approve() is the only caller of apply" has to be
           re-established at a new home, or it is lost
```

## Shape

The epic assumed option A: "`core` defines an `OperatorDecision` value type that
only `chat` can mint... No new package edge and no Protocol port". That
assumption was made before `core` was proven domain-free by
`test_core_is_domain_free`, and before the type was written carrying an `Actor`.

`OperatorDecision` is domain-shaped - it is about conversations, events and
actors. Putting it in `core` either drags `Actor` in with it, or strips the
actor field and makes the capability less useful to the consumer that wants to
name who approved.

That is not a refutation of A, but it is a real cost the epic did not price, and
it is why this is a task rather than a line in the decoupling task.

## Consequences and open questions

- **Open, and the whole task:** A, B or C. My leaning is B - the type and its
  mint stay together, `core` stays domain-free, and one declared edge is a
  cheaper honesty than a domain type in the shared base. But B makes `hostctl`
  depend on `chat`, and the epic's argument for moving approval OUT of `hostctl`
  was that it "exists to talk to `hostd`". B answers that by noting hostctl
  would depend on chat for a TYPE, not own a mechanism - which is a different
  thing, but a reviewer could reasonably disagree.
- **C deserves more weight than it looks.** It is closest to the epic's stated
  intent, and it is the only option where `hostctl` genuinely shrinks. Its cost
  is the one property in this area worth protecting, so it should be rejected
  explicitly rather than by omission.
- The type has NO production caller today - its callers are tests. So this move
  is cheap now and gets more expensive with every consumer added.
