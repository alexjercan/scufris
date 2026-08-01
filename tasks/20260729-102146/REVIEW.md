# Review: Spike: inventory app-owned mutable state and reproduce the write races

- TASK: 20260729-102146
- BRANCH: master (landed 90a0288; no sprout - the deliverable is a record plus
  its evidence script, and no `scufris/` or `tests/` file is touched)

## Round 1

- REVIEWER: in-session (this session's operator rules prohibit subagent
  delegation, so the out-of-context default could not be used; compensated by
  re-deriving every load-bearing claim from the source and by re-running the
  reproduction with added instrumentation - see Verification below)
- VERDICT: REQUEST_CHANGES

- [ ] R1.1 (MAJOR) tasks/20260729-102146/SPIKE.md:200 - the headline numbers do
  not isolate the claim they are offered as proof of. "200 agents, 102 sessions
  and 67 outcomes" is presented as evidence that `mark_finished`'s three-file
  update tore apart, but `agents.json` also contains every agent whose
  `mark_finished` was never reached because `create` raised first, so most of
  the 200-vs-67 gap is skipped iterations, not lost writes. The script cannot
  tell the two apart. Instrumenting the split (below) does, and the isolated
  figure is stronger than the one quoted: of 110 agents that reached
  `mark_finished`, all 110 got a session but **45 got no outcome**. Change: have
  `scenario_agents` record which call raised and how many finished agents lack
  an outcome, and quote that isolated number in the "What each observation
  shows" bullet instead of the raw file counts.

  Response: fixed, and the finding's own number did not survive its own fix.
  `scenario_agents` now records `create_raised`, `mark_finished_called`,
  `mark_finished_raised`, `mark_finished_returned`, `called_with_session`,
  `called_session_but_no_outcome` and `returned_without_outcome`. The "45 of
  110" quoted in this finding counted agents whose `mark_finished` was CALLED
  and then raised partway; instrumenting the returned-cleanly case separately
  shows `returned_without_outcome: 0` - every call that returned did land all
  three files. The defensible isolated figure is therefore the one now in
  observation 2: of 100 agents whose `mark_finished` was called, all 100 got a
  session mapping and 35 ended with a session and no outcome. The finding's
  substance stands (the raw counts did not isolate the claim); its replacement
  number was itself not tight enough, and the record now says which raise the
  inconsistency sits behind.

- [ ] R1.2 (MAJOR) tasks/20260729-102146/TASK.md:42 - Step 5 ("enumerate the
  remaining lost-update and partial-write windows found by read-modify-write
  inspection, each with the code location that opens it") is ticked but not
  delivered. `SPIKE.md` mentions read-modify-write three times in passing
  (`:59`, `:78`, `:219`) and has no enumeration section; the DoD command only
  greps for `scufris/.*\.py:[0-9]+`, which the inventory table satisfies on its
  own, so the proof passes without the step being done. This is the output the
  successor spike most needs - it has to size the boundary against every window,
  not just the persist path. At least these are absent: `OutcomeStore.acknowledge`
  (`scufris/agent_store/outcomes.py:204`), `SettingsStore.apply`
  (`scufris/settings_store.py:152`), `SchedulerStore.get`, which persists on a
  READ path (`scufris/scheduler.py:107`), `DigestStore.mark_delivered`
  (`scufris/digest.py:202`), `SessionRegistry.add`/`remove`
  (`scufris/agent_store/registry.py:129,154`), and the `preserve_signal` check
  in `AgentStore.mark_finished` that reads an outcome and writes it back
  (`scufris/agent_store/store.py:456`). Change: add a `### Read-modify-write
  windows` subsection listing each with its location and what a concurrent
  writer costs, and tighten the DoD command to something the inventory table
  cannot satisfy by itself.

  Response: fixed. `SPIKE.md:141` is a new `### Read-modify-write windows`
  section: an eight-row table covering every location this finding named, each
  with the read site, the write site and what a concurrent writer costs, plus
  the point that a lock inside `_persist` closes none of them because the
  window opens where the state is READ. `SchedulerStore.get` is called out
  separately. The DoD proof now greps for the section heading AND for the five
  specific locations that appear only inside it, so the inventory table cannot
  satisfy it. A new Recommendation constraint 5 carries the consequence
  forward to 20260801-100405.

- [ ] R1.3 (MAJOR) tasks/20260729-102146/SPIKE.md:273 - the Recommendation
  misses a failure mode the evidence contains: a persist that raises leaves the
  in-memory store MUTATED. `ProjectStore.create` inserts at
  `scufris/projects.py:159` and only then calls `_persist` at `:160`, so a
  caller that received a 500 has a record that is live in memory and will be
  published by the next successful write. Re-running the reproduction with 8
  threads creating the same name: 92 of 200 calls raised, yet all 200 records
  were in the store afterwards. That inverts the document's framing - records
  are not only lost, they are also silently created against an error response -
  and no constraint in the Recommendation covers it. The same shape is in
  `AgentStore.create` (`scufris/agent_store/store.py:239`), `update`/`delete` in
  both stores, and `OutcomeStore.set` (`scufris/agent_store/outcomes.py:83`).
  Change: add a constraint that a failed commit must roll the in-memory state
  back, and record the observation in "What each observation shows".

  Response: fixed. `scenario_projects` now records the names whose `create`
  raised and checks how many are still live in the store: 88 of 88 in the
  committed run (`create_raised: 88`, `raised_but_live_in_memory: 88`). This is
  observation 3 in "What each observation shows", framed as the inverse of the
  rest of the record - the stores also silently commit writes reported as
  failed. Recommendation constraint 4 requires commit-or-revert mutators and
  says explicitly that this is a constraint on the store API, not only on the
  file format. Limitations notes that "published by the next successful write"
  is still read from the code path, not observed end to end.

- [ ] R1.4 (MINOR) tasks/20260729-102146/SPIKE.md:16 - the module docstring and
  the Context both name shared-tmp CORRUPTION as failure A, but the only
  corruption this spike actually observed was in the reasoning sidecar, and it
  is reported four sections later as a log line rather than as a scenario
  result. `_observe` never returned CORRUPT for any snapshot store across three
  runs. Change: state in the Context that the raise is the common outcome and
  the torn file the rarer one, and cite where each was seen, so a reader does
  not expect `file_verdict: CORRUPT` and conclude the run failed to reproduce.

  Response: fixed in the module docstring (failure A now names the raise as the
  common outcome and the torn file as the rarer one, and says where each was
  seen) and in `SPIKE.md` Context, which adds the frequencies, the reason the
  snapshot stores collide on the rename instead, and an explicit "do not read
  `file_verdict: parses` as no failure".

- [ ] R1.5 (MINOR) tasks/20260729-102146/repro_state_races.py:31 - the exit-code
  contract is inverted relative to every other check in this repository: 0 means
  "a failure was reproduced". Nothing in `AGENTS.md` runs this script, so
  nothing breaks today, but a future `nix flake check` entry or a bored operator
  piping it into `&&` gets the opposite of what the convention implies. Change:
  return 0 for a completed run and signal "nothing reproduced" on stdout plus a
  distinct non-zero code, or rename the concept in the docstring to make the
  inversion unmissable at the call site.

  Response: fixed by taking both halves. The clean-run code is now 2 rather
  than 1, so it cannot be confused with a generic failure; the docstring leads
  with "EXIT CODES ARE INVERTED relative to a test runner" and warns against
  `&&` chains; both printed summary lines state the code they exit with; and
  the SPIKE.md Reproduction section says the same. The inversion is kept rather
  than removed because reproducing a failure IS this script's success
  condition, and a script that exited non-zero on a successful reproduction
  would be the more confusing artifact.

- [ ] R1.6 (NIT) tasks/20260729-102146/SPIKE.md:145 - "Observed, run of
  2026-08-01" pins the evidence to a date but not to the machine or the commit,
  and the Limitations section says the counts are machine-dependent. Change: add
  the commit the run was taken at, so a later re-run that disagrees can be
  attributed rather than argued about.

  Response: fixed. The heading is now "Observed, at commit 54714b7 (Linux
  x86_64, 24 cores, 8 threads x 25 writes)", and Limitations lists the counts
  seen across the other runs (90/174, 93/175, 100/171 exceptions; 3 to 26
  published regressions) with the instruction to compare verdicts rather than
  counts.

Process signal: the spike's Steps and its DoD commands were written before the
evidence existed, and R1.2 is the consequence - a `rg` proof that the inventory
table satisfies incidentally let a genuinely undone step be ticked. Worth the
retro: proof commands that can be satisfied by a section other than the one the
step is about are not proofs.

Process signal: the out-of-context round-1 reviewer this skill defaults to was
unavailable under this session's rules. The compensations are recorded above,
but a spike whose whole value is that its evidence survives scrutiny is exactly
the case where a fresh reader is worth the most. If this task is re-reviewed
later, prefer a fresh context over trusting this round.

### Verification

Re-derived from source, not from the implementing session:

- Every route cited in the mutator matrix is a synchronous `def`
  (`scufris/app.py:1893,1986,2004,2029,2116,2177,3004,3031,3058`), so FastAPI's
  thread-pool dispatch claim holds and the thread-pool x thread-pool pair is
  real.
- `AgentStore.mark_finished` does write three files with no transaction
  (`scufris/agent_store/store.py:502` registry, `:503` outcomes, `:506` agents).
- `SessionStore` holds `self._lock` across every read-modify-`_flush`
  (`scufris/auth/store.py:88,109,129,148,155`), so the "already lock-protected"
  claim is correct.
- `HostActionStore` has no path and no persist method
  (`scufris/host_actions.py:182-193`); the queue is rebuilt from the helper
  (`scufris/host_approvals.py:287`). The epic-facing finding is sound.
- Re-ran the reproduction twice more. The verdicts reproduce every time; the
  counts move (90/174, 93/175, and 90/89 with the instrumented split), which
  matches what Limitations already says.

Could not verify:

- The duplicate-id window in `_unique_id` (`scufris/projects.py:127`,
  `scufris/agent_store/store.py:182`) is a check-then-insert with no lock, but 8
  threads x 25 creates of one name produced no collision - the window between
  the check and the insert is too narrow to hit from Python. Not raised as a
  finding: unreproduced, and the spike does not claim it.
- No crash injection, so R1.3's "will be published by the next successful write"
  is read from the code path, not observed end to end.
