# Notes: Retire the two open round-1 minors on the delivery contract

## What changes

Nothing executes differently. Two prose surfaces that describe the delivery
contract currently over-promise, and Lane 2's channel author reads both before
writing a real channel against them.

Before:

- `claim_delivery`'s first docstring paragraph says `False` is returned "only
  for a ``confirmed`` row" with no qualification, so the completed-delivery
  no-op reads as unconditional. The same docstring's last paragraph already
  qualifies the neighbouring `IntegrityError` claim on
  `scufris_core.engine`'s immediate begin, so the two paragraphs disagree about
  how much the engine is carrying.
- `tasks/20260804-115319/DECISION.md` section 6 still closes with "no correct
  caller reaches that, because every one gates its send on a `True` claim, and a
  `True` claim always leaves the row `claimed`". Round 2 of that task falsified
  the last clause, and `test_two_overlapping_passes_over_one_channel_both_complete`
  is the live counterexample.

After:

- The docstring's `False`-only promise carries the same immediate-begin
  assumption as its `IntegrityError` sibling, so a reader sees one contract with
  one caveat rather than a strong claim and a weak one.
- Section 6's falsified clause carries a dated pointer to
  `tasks/20260804-141639/DECISION.md`, in place, without the surrounding history
  being rewritten.

## Surfaces

| File | Why |
|-|-|
| `packages/chat/src/scufris_chat/store.py` | `claim_delivery` docstring, first paragraph (~line 233). The only live surface still making the unqualified promise. |
| `tasks/20260804-115319/DECISION.md` | Section 6 (~line 127). Superseded record whose falsified sentence is still the one a cold reader lands on. |

Nothing else. `grep -rn "no correct caller" packages/` is already empty: round 2
cleared the docstring and README copies of that claim and left only the record.

## Data and interfaces

None. No signature, type, schema or exported name changes. `claim_delivery`
stays `(Connection, str, str, int) -> bool` with identical behavior on every
input; the change is inside its docstring literal.

## Sketches

Illustrative only.

`packages/chat/src/scufris_chat/store.py`, first docstring paragraph:

```diff
-    ever confirmed - which is exactly "someone was mid-send when we died", the
-    one case a restart must retry. ``False`` only for a ``confirmed`` row, which
-    is what makes a replay of a completed delivery a no-op at the STORAGE layer.
+    ever confirmed - which is exactly "someone was mid-send when we died", the
+    one case a restart must retry. Under the immediate begin
+    ``scufris_core.engine`` opens, ``False`` only for a ``confirmed`` row, which
+    is what makes a replay of a completed delivery a no-op at the STORAGE layer;
+    under some other engine's deferred begin a claimant that read nothing and
+    lost the INSERT race answers ``True`` without re-reading, so a completed
+    delivery can be sent twice.
```

`tasks/20260804-115319/DECISION.md`, under section 6's closing sentence:

```diff
 `confirm_delivery` is the mirror: it raises unless a `claimed` row matches, and
 no correct caller reaches that, because every one gates its send on a `True`
 claim, and a `True` claim always leaves the row `claimed`.
+
+> 20260804: the last clause is false and
+> `tasks/20260804-141639/DECISION.md` supersedes it. A `True` claim on an
+> abandoned row leaves it `claimed`, but two overlapping passes can both be
+> handed that row, and the second `confirm_delivery` reaches the raise.
```

The blockquote form is what `tasks/20260801-100405/DECISION.md:25` and `:135`
already use for an in-place correction of a superseded record.

## Shape

```
  claim_delivery docstring                  115319/DECISION.md section 6
  -------------------------                 ----------------------------
  para 1: False only for confirmed  <-- 1   "no correct caller reaches that"
  para 4: ...but a deferred begin           |
          makes the conflict a no-op        2 --> append dated pointer
                                                   |
                    both read by                   v
                       Lane 2  <----- 141639/DECISION.md (the correction)
```

Two independent edits. Neither depends on the other and neither touches code
paths, so the whole change is one commit's worth of prose.

## Consequences and open questions

- Cost: none at runtime. No test exists or can exist in-tree for the docstring
  caveat, because no non-immediate engine is constructible here; the criterion
  is a side-by-side read.
- What it forecloses: nothing. If a lane ever introduces a deferred-begin
  engine, the caveat becomes the ticket for re-adding the conflict loser's
  re-SELECT rather than a surprise.
- Assumption recorded rather than asked: the DoD's grep criterion
  (`the only tasks/ hit carries its dated round-2 note`) is read as "the only
  live *claim* is the one at `115319/DECISION.md:127`". The literal grep also
  hits `20260804-141639`'s own DECISION/NOTES/RETRO/REVIEW and this task's
  TASK.md, all of which quote the sentence in order to retire it. Those are
  correct as they stand and are not edited.
- Open: none blocking. Wording of both insertions is the implementer's, subject
  to the two manual criteria.
