# Notes: scheduled host checks and the digest

What shipped and why. The four forks are in `DECISION.md`; two of its sections were
amended by what the build measured.

## What shipped

- `scufris/checks.py` - six checks, each a bounded read judged against a threshold
  from settings, returning a structured result. `ESCALATABLE` is the allowlist for
  what a threshold may propose (R2 cleanup only), enforced at construction.
- `scufris/digest.py` - the renderer and a bounded persisted store of recent digests.
- `scufris/scheduler.py` - `watch` and `daily`, persisted next-due/last-run/
  last-result, no overlap, missed windows recorded.
- `scufris/app.py` - what a run DOES (read, render, deliver, escalate), the lifespan
  task, `GET /api/host/digests`, and an operator-only `POST /api/host/digests/run`.
- the `/host/` page grew a "What has been watching" section; `examples/host_digest.py`
  prints the digest in all five states.

Three properties worth naming, because each is a way the feature could have become a
nuisance instead of a help:

- **Silence means something.** `watch` speaks only when something CHANGED; `daily`
  always speaks, even when it is one line. The daily line is what makes `watch`'s
  silence readable as "nothing is wrong" rather than "is it even running".
- **A blank is never a pass.** UNAVAILABLE is its own state, a raise or a timeout
  becomes a NAMED failure inside the digest, and the all-clear line names any check
  that could not be read. A digest that quietly got shorter would read as good news.
- **Automation may ask, never act.** A breach proposes through the same
  `HostApprovalService` an agent uses, only for R2 cleanup, only when the state
  changed, only if no equivalent proposal is already waiting, and only when the
  operator has switched escalation on.

## The three corrections the build forced

**1. Nothing may fire on a fresh schedule.** The first version ran a pass the moment
it saw an unscheduled schedule ("so the feature proves itself"). That made every app
start - including every test that boots one - perform real subprocess reads of the
host, and made a restart loop a way to hammer the machine. The suite went from 32s to
69s and one test to 38s before this was found. A fresh schedule now arms one window
ahead and runs nothing; the operator's run-now button is how it proves itself.

**2. The run-now endpoint must not block.** It waited for the pass, so the route sweep
in `test_authenticated_session_and_csrf_boundary` (which fires every endpoint) spent
38 seconds walking the nix store. It now answers 202 and runs in the background, which
is also the right production behaviour: no HTTP request should hold a connection open
for a store walk. The client polls, exactly as it polls an approved action.

**3. A persistent condition was re-sent every interval (review R1.1/R1.2).** `watch`
rendered whenever anything wanted attention, so a 96%-full disk produced a message
every fifteen minutes - measured at 96 a day for a disk that had not moved - and with
escalation on, a fresh root-action proposal alongside each one. `render_digest` now
requires a CHANGE, and escalation requires both a change and no equivalent proposal
already pending. This is the finding that decides the manual acceptance, and it was
invisible in every single-pass test.

Reading the host is now injectable (`create_app(host_inspector=...)`) for the same
reason the NixOS build is: a real pass walks the store, so tests inject an inspector
over a `FakeRunner` replaying captured output.

## Difficulties

- **A response model defined inside `create_app` breaks `app.openapi()`**: FastAPI
  cannot resolve the forward reference of a local class ("not fully defined"). Every
  other response model in that file is module-level; `DigestView` had to join them.
- **The two whitelists must move together.** `WRITABLE_KEYS` and
  `AgentConfigUpdate`'s fields are hand-kept copies, and a test enforces that - which
  is how twelve new settings got added to both rather than one.
- **The strict tsc build caught what vitest did not**, again: adding `runChecks` to
  `HostActions` left an inline object literal in a test incomplete, and only the
  webpack build objected (`type-change-fails-strict-tsc-not-vitest`).

## Self-reflected feedback

- **A feature about attention needs a test about REPETITION.** Every test I wrote
  drove one pass, and every one passed, while the actual behaviour over an hour was
  the thing that would have got the feature muted. For anything that speaks on a
  schedule, the first test should be "what does an unchanged world produce over N
  ticks".
- **"Prove it works on first boot" was my invention, not the plan's** - and it cost
  real-host I/O in every test. When adding a convenience nobody asked for, ask what it
  does to the paths that are not the happy one (a restart loop, a test suite).
- **Reading the rendered output beat reading the code twice**: the detail-placement
  bug and the wording of the all-clear both came from running
  `examples/host_digest.py` and looking, not from review of the renderer.
