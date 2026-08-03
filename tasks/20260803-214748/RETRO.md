# Retro: Move read-only host inspection into packages/host

- TASK: 20260803-214748
- BRANCH: refactor/host-package
- REVIEW ROUNDS: 5

## What went well

The carve itself was mechanical and no round found a defect in it. The plan's
static inspection held exactly: thirteen call sites, three pairwise-disjoint
name sets, a moving tree importing nothing but stdlib, `psutil` and `pydantic`.
That is why `packages/host/pyproject.toml` could declare no `scufris-core` and
have the claim be a dependency list rather than a sentence.

`test_no_package_imports_a_sibling_private_module` did the job it was written
for on its first real run, and was falsified before being trusted: inserting
`from scufris_host.run import Runner` into `scufris/checks.py` turns it RED.
The epic's one rule is now enforced by a check that has bitten, not by a README
paragraph.

The two proofs the plan invented were the right two.
`test_stats_endpoint_matches_inspector_output` makes "Stats serves the same
payload" falsifiable across a move that otherwise has no behavioral assertion,
and `examples/host_report_fixture.py` turned the `FakeRunner`/`ok_result` seam
that `run.py`'s docstring had promised for exactly this into something the
suite runs.

## What went wrong

Five rounds and eleven findings, and only two were code: R1.3's `sys.path`
insert pointing at the repo root, and R4.3's unrequested `DEFAULT_CONFIG_REPO`
export. The other nine were claims ABOUT the code - four counts, three methods,
two stale paths.

**The counts.** The plan hand-counted renderers (seventeen, actually fourteen),
`__all__` additions (four, then five, then four again) and metrics' public names
(fifteen, actually sixteen). Each wrong number was then copied forward into
DECISION.md and the close-out, so one hand-count seeded three surfaces. R1.4
caught a miscount inside a correction to a miscount. The one count that was
right all along - thirteen call sites - came from a grep.

**The methods.** R1.3's fix was recorded as confirmed under `python -P`. It was
not: `-P` drops cwd and the script directory but leaves `site` running, and
`site` processes `_editable_impl_scufris_host.pth`, whose one line is
`<worktree>/packages/host/src`. So `scufris_host` imported either way and the
run distinguished nothing - the broken insert had never been falsified. Round
3's correction then reached for `-S -P`, which does bypass the `.pth` but drops
site-packages wholesale, so the example dies on `pydantic` before rendering
anything. Two rounds went to a sentence about how something was checked.

**The sweeps.** R1.1, R4.2 and R5.1 are one defect in three surfaces: a doc
line naming a path that moved. R5.1 is the sharpest - four rounds of sweeping
found nothing because every sweep grepped the dotted `scufris.host`, and
`CHANGELOG.md:111` spells it `scufris/host/README.md`.

R5.1 is still open. It is a NIT and a one-token edit, deliberately not fixed
after the APPROVE rather than reopening a sixth round for it.

## What to improve next time

**Breadth.** 85 files and +1663/-317, and the diff is the right size: ~20 root
test modules and thirteen call sites all re-point in one commit because a
distribution boundary cannot move halfway. No split was missed. The one place
scope grew past the plan was found by doing rather than reading -
`HostInspector`/`HostOverview` had to move to `inspector.py` because
`overview.py` imports them from the package root, so adding
`from .overview import ...` to the facade is a cycle that fails at runtime. A
plan step reading "re-export X from `__init__`" is worth checking against
whether X's module imports `__init__` back; a static read cannot see it.

**Churn.** No round-1 finding was a design finding, so the from-scratch
challenge would have changed nothing. The plan-time question that would have
prevented nine of eleven findings is narrower and not currently asked anywhere:
*which numbers in this plan came from a command, and which from counting?* The
plan wrote five counts as prose. A count is a claim about code that shares no
token with the code it describes, so no grep, linter or doc sweep will ever
find it stale - the only defence is to not write one without the command that
produced it beside it. DECISION.md now carries its derivation for that reason.

The sweep lesson is the same shape one level up: a doc sweep is only as
complete as the spellings it searches. Renaming `scufris.host` ->
`scufris_host` means sweeping the dotted form, the slash form and the bare
directory name, and writing down which were covered. Four rounds each swept one
form and each believed it was done.

**Context.** One observed pressure point, and it was environmental rather than
token-driven: the `import scufris_host` proof reads RED in a worktree whose
`.venv` predates the new member, and `uv sync` in the worktree fixes it. Worth
knowing before treating that exit 1 as a regression. The flow also crossed a
context cut between rounds 4 and 5 with no loss, because REVIEW.md carried the
open findings and their evidence - the record did the handoff, not a summary.

## Action items

- R5.1 (`CHANGELOG.md:111` -> `packages/host/src/scufris_host/README.md`)
  remains open and unfixed. Fold it into the next task that touches CHANGELOG,
  or take it as a one-line edit before the v0.2.0 release notes ship.
- `tests/test_app.py::test_orchestrator_chat_uses_server_cwd` failed one
  full-suite run mid-branch and passed alone and on the next full run of the
  same tree. Pre-existing and untouched here; filed as 20260804-003731.
- `hostd` and `hostctl` carve against a facade that already exports what they
  use, so their tasks do not reopen the facade question - but they inherit the
  flat namespace, and a name added to `metrics` or `processes` that collides
  with a report name will be a visible `__all__` edit rather than a silent
  shadow. That is the whole safety mechanism; do not reintroduce addressable
  submodules to work around a collision.

## Landing message

```
refactor(host): carve read-only host inspection into packages/host

`scufris/host/`, `scufris/metrics.py` and `scufris/processes.py` become the
`scufris-host` distribution at `packages/host`, import root `scufris_host`.
It declares `psutil` and `pydantic` and nothing else - not even
`scufris-core` - so the epic's `host -> nothing` edge is a dependency list
rather than a claim, and `psutil` leaves the root's dependencies.

Every consumer goes through the widened facade: the thirteen root modules that
reached into `host.run`, `.models`, `.storage`, `.units` and `.overview` now
import from `scufris_host`, which is the first run of
`test_no_package_imports_a_sibling_private_module` with something to police.
`HostInspector`/`HostOverview` moved to `inspector.py` to break the import
cycle that widening the facade creates.

No behavior change: the same wheel, the same endpoints, the same payloads.
`test_stats_endpoint_matches_inspector_output` pins that, and
`examples/host_report_fixture.py` renders every report from canned fixtures
with no host at all.
```
