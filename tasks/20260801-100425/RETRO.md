# Retro: Characterize app routes and extract the auth and host routers

- TASK: 20260801-100425
- BRANCH: refactor/extract-auth-host-routers
- REVIEW ROUNDS: 2

## What went well

Step 1 shipped alone (`5bb67e4`, no `app.py` edit) and the expected route-table
literal was never touched again, so every later commit was judged against a
contract written before the surgery. Both DoD tests were falsified before being
trusted; TASK.md Reflection records how.

## What went wrong

- **A `cmd:` proof named a command that has never existed.** The DoD carried
  `npm run test:e2e`; `web/package.json` has no such script. It was corrected to
  `npm run ci` mid-task and disclosed. The plan gate wants `cmd:` proofs red on
  base, and this one was red - but red for the wrong reason (missing script, not
  missing behaviour), and nobody read the failure text.
- **A compound DoD clause got one proof.** "The auth **and** host routers can be
  tested with fake services" was proven by a rig that built only the two host
  routers. Half the claim shipped unproven until R1.2. The conjunction was in the
  plan; the single test id next to it was too.
- **Evidence was carried, not re-derived.** The Evidence table recorded a pass
  count no commit on the branch ever produced (R1.1), because it was copied from
  an earlier run instead of re-run after the last commit.
- **The round-1 fix for a fail-open guard shipped unfalsified** (R2.2). R1.3
  closed `iter_routes`'s silent skip with a `raise TypeError` that has no test -
  the same "a guard is not a guard until it has been falsified" lesson the
  record's own Reflection draws, recurring on the fix that drew it.

## What to improve next time

- **Breadth.** ~860 lines left `app.py` across four seams (SSE, auth, host
  services, host routers). Step 1 was correctly split out and landed alone. The
  remaining bundle is a weak boundary rather than an inherently large feature:
  the auth extraction (`api/sse.py`, `api/auth.py`) and the host extraction
  (`host/overview.py`, the three services, `api/host.py`, `api/hostconfig.py`)
  share only "leaves `create_app`" and were independently landable behind the
  same characterization test. Two tasks would have given round 1 half the surface
  to hold at once.
- **Churn.** Both round-1 findings above the NIT line were plan defects, not
  worker defects, and one plan-time habit catches both: execute every `cmd:`
  proof at plan time and read the failure text, and give each conjunct of a DoD
  clause its own named proof. `plan`'s from-scratch challenge would have caught
  neither.
- **Context.** No context pressure is recorded anywhere in this task - no
  checkpoint commit, no compaction handoff, no `resume` cycle. Review ran
  out-of-context in both rounds, which is where the two findings above the NIT
  line came from. Nothing to change.

## Action items

- Five review findings are APPROVEd-but-open: R2.1 (speculative `WebSocketRoute`
  skip re-opens the fail-open hole), R2.2 (the `iter_routes` guard has no test),
  R2.3 (stale 2924/22 numbers in DECISION.md, corrected to 2923/25 in TASK.md),
  R2.4 and R2.5 (fake docstring and unread `AuthRig` fields). MINOR/NIT, so they
  did not block the verdict; the reviewer asked for them before landing. Fold in
  or drop them deliberately at the landing gate.
- Reusable observations submitted through `knowledge`; slugs in the flow output.
- `20260801-100441` and `20260729-103712` follow this shape; TASK.md Next says
  how.
