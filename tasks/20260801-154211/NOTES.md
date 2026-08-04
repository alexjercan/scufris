# Notes: Plan and release v0.2.0: Project as the daily workspace

Understanding round of 2026-08-04, prompted by the maintainer's restatement of
the plan and a new proposal: a Python SDK wrapper around `tatr`.

This task is a PLANNING task. It ships no feature. Its work product is the set
of child tasks and the order they run in, so "what changes" below is the shape
of the sprint, not the shape of a diff. The one exception is the tatr SDK,
which is a real code unit that this round places in the tree.

## What changes

### The maintainer's restatement, checked against the tree

> Create some packages, like chat, agents, etc. Build a Python SDK wrapper
> around tatr, focused on making it easy for scufris to work, not a generic
> tatr.py. After all is done and we have a nice working UI like in the
> architecture we also add Telegram. Then delete the legacy stuff and do a
> review pass to make sure all the docs and code is clean.

Four of the six claims hold as stated. Two are half-right, and both halves that
are wrong are ordering constraints that break the app if taken literally.

| Claim | Verdict |
|-|-|
| Create `chat`, `agents`, `flow` packages | Holds. Steps 3 and 5 |
| A scufris-shaped tatr SDK, not a generic one | Holds, and the architecture already names it. See below |
| UI follows the architecture/mockup | Holds. Step 7 |
| "Then we add Telegram" | HALF. Telegram needs two separate tasks, one of them EARLY |
| "Then we delete the legacy stuff" | HALF. Deletion is three tasks at three different times, not one at the end |
| "A review pass so docs and code are clean" | Holds, and it is MISSING from `TASK.md` Steps today |

### The tatr SDK is not a new idea; it is an unnamed one

`tasks/20260801-154211/architecture.html` lists the v0.2.0 foundation as five
items. The second is:

> **Structured tatr read** - Typed lifecycle metadata and safe artifact index.

So the target architecture already requires it. What is missing is a package to
put it in and a name for it. Today the requirement is buried inside
`20260729-102158`, whose Steps 2 and 3 say, in an API-route task:

> Replace parsing of display-oriented `tatr ls` lines with a structured tatr
> interface or a documented Markdown parser with explicit schema validation and
> timeout/error behavior.
>
> Put parsing and validation behind a typed project-task reader that both API
> serializers and future server-side workflow launch guards can call; routes
> must not become the only owner of lifecycle truth.

That is the SDK, described as a refactor of a route. Naming it as a unit is the
substantive change this round makes to the plan.

**The defect it fixes is already documented in the code.** `scufris/projects.py`
is the only tatr caller in the tree. It shells out to `tatr ls` and scrapes the
result with a regex (`:52-55`), and the comment above that regex records what
happened the last time tatr's output moved:

> The fields BETWEEN priority and tags are skipped rather than named: tatr has
> already added `KIND` and `FLOW STEP` there since this was written, and pinning
> the exact field list is what made every task silently disappear from the
> Projects page instead of failing loudly.

Silent total data loss on an upstream format change, in the module the whole
Project workspace reads from. The current mitigation is to match less of the
line, which lowers the chance of the failure without changing its mode.

**What "scufris-shaped, not generic" means concretely.** A generic `tatr.py`
would mirror the CLI: one function per subcommand, faithful argument passing,
`tatr rm` included. The scufris-shaped one exposes only what the architecture's
screens and the flow guard ask for, and it is asymmetric on purpose - a wide
read surface and a deliberately narrow write surface, because
`tasks/20260729-220835/DECISION.md` section 5 makes tatr authoritative and
Scufris a reader that asks permission:

> Scufris stores assignments and observations, never lifecycle truth. Before
> every launch or transition, one server-side guard: re-read the task through
> the typed reader; probe legality with `tatr flow -n <id>` [...] and on refusal
> return the REASON.

So the SDK's job is to make that one paragraph cheap to write correctly. Not to
be complete.

### Reading the CLI is less bad than `projects.py` suggests

Worth recording, because it changes how much the SDK has to invent. Of the nine
subcommands scufris needs, eight already emit tab-separated records. Verified
against `tatr 1.0.1`:

```
$ tatr frontier 20260803-213242
BLOCKED<TAB>20260804-053002<TAB>p100<TAB>WORKING+PLAN<TAB>Prove the declared ...

$ tatr proofs 20260804-053002
1<TAB>cmd<TAB>`uv run pytest -q "tests/...::test_..."`

$ tatr context 20260804-053002 --phase plan
/home/alex/personal/scufris/tasks/20260804-053002/TASK.md<TAB>present
/home/alex/personal/scufris/tasks/20260804-053002/SPIKE.md<TAB>missing
```

`tatr ls` is the outlier - `<path>: [PRIORITY: N, ...] Title` - and it is
exactly the one that broke. `tatr show` prints the whole record verbatim, so
for anything `ls` will not give up, reading `tasks/<id>/TASK.md` off disk is
both cheaper and more stable than shelling out. The SDK is therefore a HYBRID
and should say so: parse the file for record fields, shell out only for the
things tatr computes rather than stores - legality (`flow -n`), the frontier,
the proof list, the phase context, and the lint.

There is no `--json` on any subcommand. If one ever lands, the SDK is the one
place that changes.

### Telegram: two tasks, not one, and the first is early

"Then we add Telegram" reads as one task at the end. It cannot be, and the
reason is mechanical rather than a matter of taste.

`scufris/telegram/wiring.py` imports at MODULE scope (`:37-47`):

```python
from ..agent_diagnostics import (...)
from ..agent_store import ORCHESTRATOR_ID, AgentStore
from ..health import AgentHealth
from ..orchestrator import OrchestratorTurnService
```

Every one of those is legacy that step 8 deletes. Python resolves module-scope
imports at import time, so on the day the orchestrator is deleted Telegram does
not lose a feature - it raises `ImportError` and the bot does not start. The
maintainer's own constraint from the sprint cleanup was that basic conversation
must keep working and agent operations should answer "running agents is not
enabled on telegram". A module that will not import cannot answer anything.

So: a REDUCTION task, before any deletion, that cuts those four imports and
leaves Telegram talking to `chat` only; and a RECONNECTION task at the very
end. Both are already unchecked bullets in `TASK.md`; this round records WHY
the first cannot move later.

### Deletion: three moments, not one

"Then we delete the legacy stuff" is one step in the restatement and three in
reality.

1. **Already done** - `20260803-214750` removed the `/api/agent/*` router and
   the JSON import path, and squashed the migrations to one baseline. Neither
   had a replacement to wait for.
2. **Forced to interleave** - `agents` and `flow` cannot be built alongside the
   code they replace. The carve mandates one `Base` and one metadata, and
   `scufris/db/models.py` already holds `projects` (`:57`), `agents` (`:81`),
   `agent_session` (`:112`), `agent_session_history` (`:134`) and
   `agent_outcome` (`:150`). Two classes with the same `__tablename__` on one
   `DeclarativeBase` raise `InvalidRequestError` at IMPORT, so the app would not
   start with both present. The old rows come out in the same task that lands
   the new ones.
3. **The end** - the orchestrator stack and the pages that render it, once the
   replacement is live.

`chat` is the only package genuinely free to grow alongside: `conversation`,
`event`, `delivery` and `activity` collide with nothing in the ten table names
above.

Roughly 5,600 lines across `orchestrator/`, `sessions/`, `agent_store/`,
`agent/` and `backends/` are in the eventual blast radius.

### The cleanup pass is real and currently unwritten

`TASK.md` Steps end with "run the canonical gates" and "cut and publish". There
is no step for the sweep the maintainer describes - stale docs, dead comments,
`README.md` sections describing deleted surfaces, `CHANGELOG.md`. Given that
this release deletes about a third of the application, that sweep is a task,
not a checklist item folded into the last commit.

## Surfaces

Files this understanding round read and reasoned about. Only `NOTES.md` and
later `TASK.md` change in this task; the rest are evidence.

| Path | Why |
|-|-|
| `tasks/20260801-154211/TASK.md` | the sprint order and the twelve unminted task bullets |
| `tasks/20260801-154211/architecture.html` | target architecture; names "Structured tatr read" as v0.2.0 foundation |
| `tasks/20260729-220835/DECISION.md` | section 5 fixes tatr as authority and defines the guard the SDK serves |
| `tasks/20260729-102157/TASK.md` | the product epic; Done Means 1-3 are the SDK's consumers |
| `tasks/20260729-102158/TASK.md` | holds the SDK requirement today, framed as a route refactor |
| `tasks/20260803-213242/TASK.md` | the ten-unit table; decides where the SDK lives |
| `scufris/projects.py:43-55, 265-300` | the entire current tatr integration; the regex and its incident comment |
| `scufris/project_capabilities.py:116-225` | second tasks-dir reader; walks the directory, does not call tatr |
| `scufris/db/models.py:57-150` | the five table names that force delete-then-build |
| `scufris/telegram/wiring.py:37-47` | the four module-scope imports that make deletion an ImportError |
| `scufris/mcp_server.py:26-27` | records that tatr tools are deliberately absent from MCP |
| `tests/test_examples.py:32-38` | the `OFFLINE` opt-in tuple every new package must join |

## Data and interfaces

### Where the SDK lives

The epic's ten-unit table gives `packages/flow` ownership of "the tatr reader,
the flow guard, assignments, projects". The reader is already assigned. So the
SDK is a module inside `flow`:

```
packages/flow/src/scufris_flow/
    tatr/
        __init__.py     # the public surface below
        cli.py          # subprocess: which(), timeout, exit code, stderr
        record.py       # TASK.md -> TaskRecord
        errors.py       # TatrMissing, TatrTimeout, TatrFailed, TatrUnparsable
    guard.py            # the consumer from DECISION.md section 5
```

Not its own package, on the KISS rule in `AGENTS.md` - one caller is not an
abstraction, and `flow` is the only caller the architecture names. Recorded as
an open question below because the argument is not airtight.

### Public surface

Illustrative signatures, to show the shape and the read/write asymmetry - not a
final API.

```python
# Read. Wide, because every screen in architecture.html needs some of it.
def read_task(tasks_dir: Path, task_id: str) -> TaskRecord: ...
def list_tasks(tasks_dir: Path, *, query: str | None = None) -> TaskList: ...
def artifacts(tasks_dir: Path, task_id: str) -> list[Artifact]: ...
def proofs(tasks_dir: Path, task_id: str) -> list[Proof]: ...
def frontier(tasks_dir: Path, epic_id: str) -> list[FrontierRow]: ...
def phase_context(tasks_dir: Path, task_id: str, phase: Phase) -> list[ContextPath]: ...

# Probe. The guard's whole reason to exist. Never mutates.
def probe_advance(tasks_dir: Path, task_id: str) -> Legality: ...

# Write. Narrow, and each one is a lifecycle transition the operator approved.
def advance(tasks_dir: Path, task_id: str) -> TaskRecord: ...
def rewind(tasks_dir: Path, task_id: str, to: Activity) -> TaskRecord: ...
```

`rm` and `migrate` are absent on purpose: destructive, and nothing in the
architecture asks for them.

```python
class TaskRecord(BaseModel):
    id: str
    title: str
    priority: int
    kind: Kind                  # TASK | EPIC | STORY | SPIKE
    activity: Activity | None   # None when the record shows "-"
    gates: frozenset[Gate]
    resolution: Resolution | None
    status: Status              # DERIVED by tatr; never written by scufris
    tags: frozenset[str]
    parent: str | None
    depends_on: tuple[str, ...]
    steps: tuple[Step, ...]     # text + checked
    artifacts: tuple[Artifact, ...]

class Legality(BaseModel):
    """`tatr flow -n` decoded. The `reason` is what the UI renders on a
    disabled control - epic 20260729-102157 Done Means 3."""
    allowed: bool
    from_activity: Activity
    to_activity: Activity | None
    gate: Gate | None
    reason: str | None
```

`Legality` is the type that turns an unexplained greyed-out button into an
explained one, which is a named Done Means of the product epic. Everything else
in the SDK exists to serve some panel in `architecture.html`.

### Two things the SDK must get right or it is not worth building

**Fail loudly.** A malformed or unrecognised record becomes an explicit
`PartialRecord` in the returned list, never a silently dropped row. This is
`20260729-102158`'s Done Means
(`test_project_tasks_report_partial_parse_failures`) and the direct fix for the
incident in the `projects.py` comment.

**Pin the version.** `tatr version` prints `tatr 1.0.1`. The SDK asserts a
supported range once at startup and fails with a named error, so the next
upstream format change is a loud refusal at boot rather than an empty board.

## Sketches

Illustrative only.

The regex disappears from `projects.py`:

```diff
-_TASK_LINE_RE = re.compile(
-    r"^(?P<path>.+?): \[PRIORITY: (?P<pri>-?\d+), (?:[^\]]*?, )?"
-    r"TAGS: (?P<tags>[^\]]*)\] (?P<title>.*)$"
-)
+from scufris_flow.tatr import list_tasks
```

The guard from DECISION.md section 5 becomes readable:

```diff
+    legality = tatr.probe_advance(project.tasks_dir, task_id)
+    if not legality.allowed:
+        return NextAction(available=False, reason=legality.reason)
+    if not conversation.has_operator_approval(task_id, legality.gate):
+        return NextAction(available=False, reason=f"{legality.gate} needs your approval")
```

Telegram's reduction is a deletion, not a rewrite:

```diff
-from ..agent_diagnostics import (...)
-from ..agent_store import ORCHESTRATOR_ID, AgentStore
-from ..health import AgentHealth
-from ..orchestrator import OrchestratorTurnService
+from scufris_chat import Conversation
```

## Shape

Where the SDK sits, and who is allowed to talk to it:

```
   architecture.html screens
   +---------------------------------------------------+
   | Task metadata | Flow state | Next legal action     |
   | Dependencies  | Artifacts  | "why not" reason      |
   +--------------------------+------------------------+
                              |
                     scufris/ (composition root, routers)
                              |
                              v
   +---------------------------------------------------+
   |  packages/flow                                     |
   |                                                    |
   |   guard.py  -- re-read, probe, require approval    |
   |      |                                             |
   |      v                                             |
   |   tatr/  <-- THE SDK. sole owner of the boundary   |
   +------|--------------------------------------|------+
          |                                      |
   read TASK.md off disk              subprocess `tatr ...`
   (record fields; stable)            (computed: flow -n,
          |                            frontier, proofs,
          v                            context, check)
   tasks/<id>/*.md                             |
                                               v
                                        tatr 1.0.1 binary
```

Sprint order, with the two constraints that fix it. `=` is a hard ordering
edge, not a preference:

```
  [1] carve packages ...................... DONE (20260803-213242, 1 child open)
  [2] delete safe half .................... DONE (20260803-214750)
       |
  [3] chat  (alongside; no table collision)
       |
  [4] telegram REDUCTION =================== must precede [8]; ImportError
       |                                     (wiring.py:37-47)
  [5] tatr SDK -> agents -> flow
       |         \____ delete-then-build; table names collide
       |               (db/models.py:57-150)
  [6] merge into the composition root
       |
  [7] UI from the mockup
       |
  [8] delete what remains
       |
  [9] telegram RECONNECT
       |
 [10] docs and code cleanup sweep .......... currently unwritten
```

## Consequences and open questions

### What this costs

The SDK is one more unit before the UI can be trusted, and it lands on the
critical path - the flow guard, the Project board and the task detail panel all
read through it. It is not large, but it is not free, and it delays the first
screen.

It also freezes a coupling: Scufris gains a hard runtime dependency on a `tatr`
binary of a supported version, and says so out loud instead of degrading to an
empty list. That is the intended trade, and it is a real operational
constraint - a Nix closure without `tatr` becomes a boot failure rather than a
blank page.

### What it forecloses

An in-process reimplementation of tatr's state machine. Deliberately - see
DECISION.md section 7, which upholds the rejection of a second workflow engine
and permits only a coordinator that "asks and renders".

### Open questions

1. **Is the SDK a module in `flow`, or `packages/tatr`?** Recommended: module
   in `flow`, per the epic's ten-unit table and KISS. The counter-argument is
   real - it is the only unit in the tree that shells out to an external binary,
   its tests need fixture repositories and a version pin, and `agents` may
   eventually want to read a task. If it grows a second package as a consumer,
   promote it then.
   - I agree with `flow` since it keeps it simple

2. **Does `20260729-102158` absorb the SDK, or does the SDK land first and
   `102158` become its route-side consumer?** Splitting is cleaner - the SDK is
   TDD-able with fixture task directories and no HTTP - but `102158` is already
   at PLANNING with a written plan. Splitting means rewinding it.
   - I think it can absorb the SDK and still do TDD; I don't think we need to
     rewind it because the PLAN gate is not yet earned

3. **Where does the write path get its authority?** `advance()` mutates the
   authoritative record. Nothing in the plan yet says which layer may call it,
   or how an operator approval event is proven to the SDK rather than to the
   guard above it. Related to the unminted STOP-GATE CONTRACT task, which is
   the harder half of the same problem.
   - give me options and explain this one

4. **Does `project_capabilities.py` move behind the SDK too?** It reads a
   project directory without calling tatr (`:116-225`). Same boundary concern -
   traversal, symlink escape, size caps - and `102158` already carries those as
   Done Means. Probably yes; not decided.
   - honestly I don't like that we have this old epic 20260729-102157 with only
     a subtask in this sprint 20260729-102158; I would say move them to backlog
     and let's create a real task for tatr + the SDK stuff; probably makes
     sense to have a `flow` package for project and task related activities,
     what do you think present me some optins and explain how this would work

5. **Still unhomed from the spike.** `DECISION.md` "Not addressed here" defers
   six questions to "v0.3.0 tasks": retention policy, summary versioning,
   per-turn event granularity, eager-versus-lazy re-seed on backend switch, the
   `SCUFRIS_ORCH_SESSION_ID` rename, and where the guard service lives. The
   last one collides with question 3 above and cannot wait for v0.3.0 - the
   flow guard is v0.2.0 work.
   - give me some options and explanation here, I don't see it yet

6. **Host approvals as conversation events** - open in `20260803-213242` and
   unchanged by this round. Under the declared dependency graph `hostctl`
   cannot reach `chat`, so DECISION.md section 6's "an approval decided from
   either channel writes ONE decision event" has no legal path. Blocks the
   `packages/telegram` carve, not `chat`.
   - also explain this one better, when I say explain, please do like ASCII
     diagrams or something I really do not understand how this is an issue
     without seeing a flow or something for me it makes sense that "user
     presses button -> hostctl get a "ok" and proceeds with whatever it was
     doing", so I don't see the issue unless you actually present it to me
