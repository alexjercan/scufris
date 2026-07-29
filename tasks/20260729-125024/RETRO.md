# Retro: expand read-only host inspection beyond stats

- TASK: 20260729-125024
- EPIC: 20260729-124655
- REVIEW ROUNDS: 3 (REQUEST_CHANGES, REQUEST_CHANGES, APPROVE)
- FINDINGS: 14 in round 1 (3 MAJOR), 2 + 6 NITs in round 2

## What went well

**Re-probing the host at plan time changed the plan three times.** The spike had
already measured this ground, and I re-ran it anyway. That found `iptables -L` is
root-only (so the task's "current firewall rule state" could not ship as
written), that the machine is a desktop rather than the laptop the task assumed,
and that the closure-diff trap reproduces exactly. All three became design
decisions BEFORE code, not discoveries during it. The lesson
`probe-runtime-on-target-host-early` earned its keep for the fourth recorded
time.

**The one-runner-seam architecture held up under review.** Both reviewers said so
independently, and it is why the degradation tests could be written at all: four
failure modes across eight reports go through one `FakeRunner`, with no
subprocess patching anywhere. Making the honesty property structural rather than
a habit was the right call - every place it broke was OUTSIDE that layer.

**The example script paid for itself before it was committed.** Three bugs on
its first live run, one of which (`psutil.SOCK_STREAM` does not exist) no faked
test would ever have caught.

**Round 2 was worth running.** I had pre-written an APPROVE for round 2 before
verifying it - and the independent pass came back REQUEST_CHANGES with a real
behavioural regression caused by one of my round-1 fixes. If I had shipped my own
verdict, a fix for a security nit would have quietly reintroduced the exact
confusion the package exists to prevent.

## What went wrong

**A fix for one finding reintroduced the bug class the package is about.**
Round 1 MINOR 10 was correct: `nix-collect-garbage --delete-older-than` has a
generation-deleting half, so the read-only guarantee rested on nix's behaviour
rather than on my code. I swapped it for `nix-store --gc --print-dead` - and kept
the OLD command's output-shape assumptions. The new command prints bare paths and
no summary line, so an empty (healthy) store would have reported "no dead-path
count was reported": empty rendered as broken, in the package built to stop
exactly that. I changed the command and did not re-measure its output.

**The frontend did not inherit the backend's discipline.** I spent the whole
backend enforcing "a blank never means fine", then wrote host cards that piped
machine-controlled strings into `innerHTML` - repeating a ledger lesson verbatim,
including the hostile-input test it prescribes. Crossing a language boundary
apparently reset my attention: I was thinking about the honesty property (which I
did carry across) and not about the escaping property (which I did not).

**`shell=False` felt like the end of the argument-safety question.** It is not:
option injection is a separate axis, and a model-supplied unit pattern of
`-Hattacker@host` makes systemctl open an outbound SSH connection with the
service user's credentials. In a package whose entire premise is "reading the
host cannot do anything", that is the worst possible defect, and I did not think
about it once while writing nine modules of argv construction.

**Three of my own tests were shaped so they could not fail** - a conditional
assertion behind `if "unavailable" not in out`, a tautology over a title every
render path emits, and `assert x or not y` inside a DoD proof. `dod-named-tests-
deserve-the-most-scrutiny` is in the ledger and I still wrote all three, in the
tests a Definition of Done points at.

**A piped gate read as green while a test was failing.** I ran
`nix flake check 2>&1 | tail -3; echo $?` and read the 0 - which was `tail`'s
exit status. The repo's own AGENTS.md warns about exactly this, and the ledger
has `nix-develop-pytest-pipe-eats-the-summary`. The failing test was real (a
sandbox-only environment difference), so the habit cost a near-miss on landing
red. Run the gate bare, or redirect to /dev/null and read `$?`.

**A DoD test asserted the environment rather than the property.** It required
`thermal.battery.ok`, which is true on this desktop (empty
`/sys/class/power_supply` -> "no battery") and false in the nix sandbox (no such
directory -> "unreadable"). Both are correct degradations. Only the sandbox
caught it, because only the sandbox has a different `/sys`.

## What to do differently

1. **Re-measure a command's output when you change the command.** A swap made for
   a safety reason is still a swap: the parser downstream was written against a
   different program. One live run would have caught it, and I had already run
   the example against the real host several times by then.
2. **Carry the invariant across the language boundary explicitly.** When a
   backend enforces a property (bounded, honest, escaped), list what the frontend
   consuming it must also enforce, and check that list - do not assume the
   property travels with the data.
3. **Treat "does this argument become a flag" as a standing question for every
   argv, separate from shell safety.** `shell=False` answers one question only.
4. **Write the NEGATIVE assertion first in a regression pin.** The store-path
   regex test initially asserted only that the good input parsed, which passed
   against the buggy pattern too; it became a real pin only when it asserted the
   pattern must REFUSE the path form. Same for every honesty test: assert that
   the wrong rendering does NOT appear, not just that the right one does.
5. **Read a gate's EXIT CODE, never a piped tail.** `| tail; echo $?` reports
   tail's status. Run it bare, or `>/dev/null 2>&1; echo $?`.
6. **When several outcomes are all correct, assert the invariant they share** -
   not whichever one the dev box happens to produce. And remember the nix sandbox
   is a different environment: anything touching `/sys`, `/proc` or PATH needs a
   test written for both.
7. **Never pre-write a verdict.** I drafted round 2's APPROVE before running
   round 2. The habit is what produced three unfalsifiable tests as well - the
   same impulse to record the conclusion I expected rather than the one I had.

## Lessons for the ledger

- `re-measure-output-when-you-swap-the-command` (Monitoring/collector)
- `backend-invariants-do-not-cross-into-the-frontend` (Frontend)
- `shell-false-does-not-stop-option-injection` (Backend/security)
- `assert-the-wrong-rendering-is-absent-not-just-the-right-one` (Testing)
- `assert-the-property-not-the-environments-answer` (Testing)
