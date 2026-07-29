# Retro: Spike - the host capability privilege and safety model

- TASK: 20260729-125020
- BRANCH: spike/host-privilege-model
- REVIEW ROUNDS: 2

## What went well

- Probing the host before writing changed the ANSWER, not just its evidence.
  Three measured facts each moved a decision: `alex` is in the `docker` group
  (so the threat model had to be rewritten from "stops an attacker" to "stops
  the model, the injection, and the unrecorded action"), every read-only
  inspection works unprivileged (so 20260729-125024 became parallelizable
  instead of blocked), and `nvd` is absent while `nix store diff-closures` is
  builtin (so the preview adds no dependency). This is
  `probe-runtime-on-target-host-early` (x3, already pending promotion) paying
  off in the one place it is easiest to skip - a docs-only task where nothing
  forces you to run anything.
- Taking the three forks to the operator WITH the constraint that made the
  options exclusive, rather than after building against a guess.
- The review earned its keep on a diff with no code in it. Two MAJORs on a
  pure-prose change is the argument against treating "docs-only" as trivial.

## What went wrong

- R1.1, the serious one. I wrote that the machine token "deliberately does not
  satisfy" the approval gate. Root cause: I established it from
  `tasks/20260729-125015/DECISION.md` and `scufris/auth.py`'s module docstring
  - both of which describe the intended design - and never opened the
  enforcement point. `scufris/app.py:840-844` short-circuits on a valid bearer
  token before the session and CSRF checks, so the opposite is true today. The
  trap is specific and worth naming: a decision record and an implementation
  read identically in prose, and the auth task's record was unusually good,
  which made trusting it feel like diligence rather than the shortcut it was.
  A false safety property in a decision record is worse than an absent one,
  because the next task inherits it as settled.
- R1.2. I ticked "Decide audit storage **and retention**" having decided only
  storage. Root cause: I checked the step off against the section I had written
  (section 7, "Audit storage"), not against the step's own text. Compound
  steps - "X and Y" - are exactly where a heading-shaped self-check fails.
- R1.3 and R1.5. I wrote the R2 preview as "bytes freed, generations removed"
  and only found out at review time that the dry-run prints a path count and
  nothing else; likewise that an unchanged closure diff prints absolutely
  nothing. Root cause: I described what the tools ought to print. Both are
  claims a single command would have settled - and I had already run neighbours
  of both commands during the research, so the cost of checking was near zero.
  The irony is direct: the doc's own thesis is that adjacent information must
  not be dressed up as a preview, and its preview column did that.

## What to improve next time

- When a document asserts that a control HOLDS, open the code that enforces it
  and cite the file:line. Intent lives in decision records; enforcement lives
  in middleware, guards and validators, and only the latter proves a property.
  If the enforcement point cannot be cited, write the claim as a requirement on
  the implementing task instead - which is exactly the shape R1.1 was fixed to.
- Tick a step by re-reading the step, not by recognising the section you wrote.
  On a compound step, resolve each conjunct separately.
- Any sentence describing a command's output is a claim to run once before it
  is written down. This is cheap during a spike, where the terminal is already
  open.

## Action items

- [x] Ledger: `enforcement-point-not-the-decision-record` (new).
- [x] Ledger: `tick-a-compound-step-conjunct-by-conjunct` (new, `-> work skill`).
- [x] Ledger: `run-the-command-before-documenting-its-output` (new).
- No follow-up code tasks. The spike's corrections were carried into
  20260729-125029 and 20260729-125035 as Notes rather than new tasks, since
  both are unstarted and will be planned against those notes.
- Raised to the operator, not filed here: `alex` being in the `docker` group is
  a `nix.dotfiles` concern, recorded on the epic's Notes as out of scope.
