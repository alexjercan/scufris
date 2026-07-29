# Retro: gate the dashboard behind an authenticated session

- TASK: 20260729-125015
- BRANCH: feature/dashboard-auth
- REVIEW ROUNDS: 2 (round 1 REQUEST_CHANGES with 10 findings, round 2 APPROVE)

## What went well

- **Reading the call graph before designing paid for itself.** Grepping for who
  actually calls `/api/*` turned up `mcp_common._api_call` - the app calls its
  own HTTP API from MCP tool subprocesses. Cookie-only auth would have broken
  `create_agent`, `run_agent`, `report_back` and every project tool, and it
  would have surfaced as a mysterious 401 in an agent turn, not as a test
  failure. That discovery reshaped the design (a second, machine identity)
  before a line was written.
- **Sabotage-to-prove-the-gate, five times before review and six after.** Every
  claim the tests make was checked by breaking the thing and watching the named
  test go red. That is what turned "the suite is green" into "the suite
  discriminates", and it caught nothing embarrassing precisely because it was
  done deliberately rather than assumed.
- **Deny-by-default plus an enumerate-the-routes test.** The DoD asked for a
  sweep instead of a hand-written list, and that was the right shape: it caught
  a simulated ungated route family by name, and it means the six later children
  of this epic get their routes gated by existing rather than by remembering.
- **The out-of-context reviewer earned its cost.** It found two real defects I
  would not have found by re-reading my own diff, and - as valuable - it
  recorded a long list of attacks that do NOT work, which is now the durable
  evidence that the boundary holds.

## What went wrong

- **R1.2 (the machine token was ambient in every child process) is the one that
  matters.** I exported `SCUFRIS_API_TOKEN` to `os.environ` so the in-process
  tool console could reach it, and `agent._codex_env` returns `dict(os.environ)`
  - so the agent CLI, every shell command the model runs under it, and every
  sub-agent held the operator's full-privilege API credential. Root cause: I
  reached for the environment because it was the carrier already in use for
  `SCUFRIS_API_BASE`, without asking who ELSE reads that environment. A
  credential is not configuration; it needed a carrier scoped to its consumers
  from the start (it now rides `Settings` and a `ContextVar`).
- **My own test made the leak invisible.** `test_agent_env_carries_the_machine_token`
  asserted the den server's declared MCP env dict lacked the token - which was
  true and irrelevant, because the leak was through inheritance, not the dict.
  Root cause: I asserted against the structure I had just written instead of
  against the thing that actually reaches the subprocess. The rewritten test
  seeds the variable in `os.environ` first, so it fails if inheritance leaks.
- **R1.1 (500 on a non-ASCII credential header) came from not asking what the
  library does with hostile input.** `hmac.compare_digest` raises `TypeError` on
  non-ASCII `str`; headers are latin-1 decoded, so an unauthenticated caller
  could make the enforcement point throw. I had reached for `compare_digest`
  for its constant-time property and never asked about its domain.
- **`git checkout <file>` ate a round of fixes.** After the second sabotage
  round I restored with `git checkout scufris/app.py scufris/auth.py`, which
  restores from the INDEX - and the index still held the pre-review commit, so
  every round-1 fix in those two files vanished. I had committed before the
  FIRST sabotage round and not before the second. The ledger already carries
  this exact lesson (`commit-before-sabotage-or-the-restore-eats-the-fix`); I
  followed it once and then stopped following it.
- **A leaked global stub in an existing test cost a real detour.** Roughly
  fifteen minutes went into diagnosing a failure that only appeared in
  whole-file runs, before finding that an unrelated test stubs the global `URL`
  and never restores it. My first response was a workaround (`vi.waitFor`) on
  the victim; the right fix was restoring the global at the source, and I only
  got there after instrumenting rather than theorizing.

## What to improve next time

- **When introducing a credential, enumerate its readers before choosing its
  carrier.** `os.environ` is read by every subprocess you spawn; a module global
  is read by every app in the process. Ask "who else can see this?" and let the
  answer pick the carrier.
- **Assert a leak-shaped property against the thing that actually receives the
  value**, not against the structure you just built. If the risk is
  inheritance, the test must seed the ambient source and then check the
  recipient.
- **Commit before EVERY sabotage, not before the first one.** The rule is
  per-sabotage-round; treating it as a once-per-task ritual is how the restore
  eats the work.
- **When a test fails only in a full-file run, suspect leaked global state in an
  EARLIER test before suspecting your own change**, and instrument rather than
  theorize - one `console.log` of the ambient globals answered in one run what
  three rounds of reasoning had not.

## Action items

- [x] Lessons ledger updated with the four generalizable entries below.
- [x] `commit-before-sabotage-or-the-restore-eats-the-fix` bumped to x2 - it is
      now recurring, and the bump is what makes that visible.
- [ ] Operator action before the next deploy: `scufris hash-password`, then
      `sops secrets/scufris.env`. Recorded in NOTES.md and in the nix.dotfiles
      commit; the service will refuse to start without it once the flake input
      is bumped past v0.1.0.
