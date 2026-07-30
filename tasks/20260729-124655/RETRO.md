# Retro: EPIC - Make Scufris a safe NixOS host operator

- TASK: 20260729-124655
- BRANCH: docs/epic-architecture (the close-out); 8 feature branches before it
- REVIEW ROUNDS: 21 across the eight children (72 findings: 5 BLOCKER, 14 MAJOR),
  plus 1 round on this close-out

Each child has its own RETRO.md with the per-task lessons, and they are already in
`LESSONS.md`. This one is about the EPIC: what the shape of the plan did to the
outcome, and what to do differently the next time a multi-child epic is run. What
was built is in `ARCHITECTURE.md`.

## What went well

- **The spike came before any mutating code, and it was the right gate.**
  20260729-125020 produced the privilege model (root helper, typed verbs, no sudo
  rules, no shell escape) and the operator accepted it before a line of privileged
  code existed. Every later child inherited an answer instead of re-litigating it,
  and the one child that had to re-cut its own scope (20260729-125035, "the config
  repo is a PROJECT") could do it cheaply because the privilege boundary was not in
  question. Ordering the auth child FIRST had the same effect from the other end:
  nothing could gain mutating power while the dashboard was still open on the LAN.
- **Making the safety property structural rather than checked.** The recurring move
  across the epic - the verb set IS the risk taxonomy, R4 is enforced by absence, the
  proposal registry lives in the privileged process, the audience split is physical,
  the audit has only an append path - is why the review findings were about
  credentials and rendering rather than about the contract leaking. Nothing in the
  final system asks "is this caller allowed to run this command"; the command does
  not exist to be asked about.
- **One decision seam, two surfaces, in that order.** Building the decision core
  (20260729-125040) before either surface meant the dashboard and Telegram children
  were mostly "what do we SHOW", and the test that compares the web refusal and the
  Telegram refusal sentence-for-sentence is short because there is one place either
  can come from. Doing it the other way would have produced two rule sets and a
  reconciliation task.
- **Measurement kept overturning reasoning, and the records say so.** Three findings
  that no amount of reading would have produced: the option-injection `-Hsomeone@host`
  in the READ-ONLY package, the 96-messages-a-day standing condition, and the
  confirmation rule that demanded a typed acknowledgement for every service restart.
  Two DECISION.md files were amended mid-build by measurement rather than defended.
  The VM test earned its cost twice on its own (the `--extra-experimental-features`
  gap and a test VM having no system profile generation).

## What went wrong

- **The strongest surfaces were the least verifiable, and that was known too late.**
  Three of the eight children shipped their acceptance as "needs a real phone / a
  real browser / a week of digests", and the epic closes with eight pending manual
  items. The examples (`telegram_approval.py`, `host_digest.py`, `host_action.py`)
  were the right mitigation and they are genuinely good stand-ins, but the epic never
  scheduled the deployment that would let any of them be confirmed. The two operator
  actions that gate ALL of it (the sops secret, `services.scufris-hostd.enable`) were
  identified in the first child and were still unperformed at close.
- **Same-shape bugs recurred across children rather than being caught once.** The
  dashboard queue's two MAJORs were one root cause (a poll that rebuilt the page over
  the operator's typing, and an error banner that never cleared); the scheduler's two
  MAJORs were one root cause (a standing condition re-notified, and re-escalated).
  In both cases the second was found by the same probe as the first. A per-child
  review found them; an epic-level "what did the last child's MAJOR teach the next
  one" pass would have found them earlier and cheaper.
- **A wrong pointer to a proving test survived in `AGENTS.md` until the close-out.**
  The doc claimed `tests/test_mcp_server.py` asserts there is no approve tool. That
  file tests something else entirely; the assertion is in
  `tests/test_host_mcp_server.py`. It was written during the epic and read by later
  children without anyone opening the file, which is the whole failure mode: a
  citation is a claim, and an unverified one about the test that proves the epic's
  central refusal is worse than no citation, because the next reader checks the
  wrong file and concludes nothing proves it. Found and fixed here (REVIEW.md
  round 1).
- **The container record grew into the only map, and it is not one.** By the last
  child, `TASK.md` held the deployment facts, four decision summaries, eight child
  post-mortems and eight manual items - and answering "how does this thing work"
  meant reading all of it plus 300 lines of `AGENTS.md`. That map should have existed
  from about the fourth child, when the shape stopped changing, rather than being
  written at close.

## What to do differently

1. **An epic that ships a deployed capability schedules its own deployment.** Make
   the operator actions that gate acceptance (a secret, an enable flag, a flake bump)
   an explicit child task with the other children depending on it for their manual
   items - not a note in the container. Otherwise the epic finishes "built" and its
   acceptance queue only grows.
2. **Between children, read the previous child's MAJOR findings, not just the
   ledger.** The ledger entry arrives after the retro; the finding itself is
   available immediately and is usually a SHAPE ("a poll fights the operator", "a
   standing condition re-fires") that the next surface repeats. One pass per child
   boundary.
3. **Write the architecture map when the shape settles, not at close.** Roughly at
   the halfway child. It is cheap then, it is what the remaining children need for
   orientation, and writing it is itself a review pass - this one found a wrong test
   citation and two diagram claims that did not match the code.
4. **Verify a doc's citations by running them.** See the lesson below.

## Lesson for the ledger

- `verify-a-doc-citation-by-running-the-grep` -> appended to `LESSONS.md`
  (Testing): a prose sentence naming the test/file that proves a property is a
  claim, and copying it forward propagates it. Grep for the named symbol in the
  named file before writing or repeating the citation.
