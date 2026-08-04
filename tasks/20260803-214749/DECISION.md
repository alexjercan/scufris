# Decision: the shape of the hostctl carve

- DATE: 20260804-030000
- STATUS: ACCEPTED
- TASK: 20260803-214749
- TAGS: architecture,packaging,host,testing

## Context

`hostctl` is the one child of the workspace carve that is not a pure move. Five
things had to change shape before the package could exist, and each of them is a
choice that outlives this task. They are recorded here rather than in the task's
close-out because a later reader following `RunPhase` or a test filename will
want the reason, not the history.

## Decision

**1. `RunPhase` lives in `scufris_core.supervisor`, not in a module of its own.**

It is the supervisor's phase enum. It has no other owner, every one of its
readers reads it through a `RunState`, and `scufris/enums.py` - the module it
came from - is explicitly the app's *option* enums for stringly-typed config and
API fields. Giving it `scufris_core/enums.py` would cost a second allowlist
entry in `CORE_MODULES` and invite exactly the junk drawer that allowlist exists
to prevent.

No compatibility re-export from `scufris/enums.py`. It had two live readers
(`scufris/orchestrator/runs.py` and `tests/test_enums.py`); a shim serving two
call sites is a concept the codebase has to keep explaining.

**2. The generic half of `Supervisor` moves; the agent's instantiation stays.**

`Supervisor`, `RunState`, `RunPhase`, `AgentRunStalled` and the type aliases go
to `scufris_core.supervisor`. `AgentSupervisor`, `_agent_error_event`,
`_agent_error_detail` and `agent_supervisor()` stay in `scufris/supervisor.py`,
because the event type they name (`StreamEvent`) belongs to the agent and `core`
must not learn what an agent turn is. The cut is exactly the
`# --- the agent's supervisor ---` banner the file already carried.

**3. Package test modules need globally unique basenames.**

`pytest` runs in prepend import mode with `testpaths = ["tests",
"packages/*/tests"]` and no `__init__.py` under any test directory, so a test
module's import name is its bare filename. Two files called
`test_supervisor.py` in two roots is a hard collection error, not a shadowing
subtlety.

Switching to `--import-mode=importlib` was rejected: roughly twenty root test
modules import their siblings by bare name (`from conftest import ...`, `from
test_auth import ...`), and importlib mode does not put each test directory on
`sys.path`. That is a suite-wide rewrite bought for a filename.

The same rule bites harder for `conftest.py`, and it is worth naming
separately: every `conftest.py` in a rootless test directory imports under the
bare name `conftest`, so one in `packages/*/tests/` does not merely sit beside
the root's - it WINS the name for the whole run, and the twenty-odd root modules
doing `from conftest import ...` fail collection. The first draft of the hostctl
suite had one, and fifteen root modules broke. A package suite therefore owns
its fixtures INSIDE its test module. No package has a `conftest.py`, and with
one test module per package suite none needs one.

So the split halves are renamed rather than duplicated:
`packages/core/tests/test_core_supervisor.py` and
`packages/hostctl/tests/test_config_change_service.py`. The root keeps the
original names, which is what the existing package suites already do.

**4. The three flat modules drop their `host`/`host_` prefix.**

Inside `scufris_hostctl` the prefix is noise: `scufris_hostctl.actions`,
`.approvals`, `.client`. `hostconfig/` keeps its name - it is already a package,
and renaming it buys nothing. This is the only naming change in the carve.

**5. The example drives a REAL unix socket, against a fake machine.**

The plan asked for "no real socket". `HostdClient` has no other transport - it
opens a unix connection and speaks frames, and that is the whole module - so an
example that stubbed the client would exercise a stub. What the plan was really
asking for is "nothing an operator has to install", and that is met by running
the helper in-process on a temporary socket over `FakeRunner`/`FakeExecutor`:
no root, no network, no NixOS machine, no `services.scufris-hostd`. The socket
is the one real thing, which is correct - it is what the package is for.

**6. The boundary rule has no exemptions; `env.py` imports the facade.**

Round 1 (R1.1). The first implementation had `env.py` import
`scufris_hostctl.models` - a sibling's private module - and paid for it with a
`SCHEMA_ASSEMBLY` allowlist in `test_no_package_imports_a_sibling_private_module`.
The premise was that a facade cannot register tables because the row classes are
private, and it is false: registration is an IMPORT SIDE EFFECT, and
`import scufris_hostctl` already reaches `models` through `actions` and
`hostconfig.changes`. Verified by importing the facade alone and reading
`Base.metadata.tables`. So `env.py` imports `scufris_hostctl`, and the rule this
task exists to install ships with zero holes.
`test_every_package_model_is_registered` is unaffected - it reads `env.py`'s
imports with `ast` and imports whatever it finds, so a dropped import still
fails it.

**7. `test_every_package_model_is_registered` builds its metadata in a
SUBPROCESS.**

Round 2 (R2.1). Importing `env.py`'s module list into the running interpreter
measures nothing under a full `pytest`: `scufris_hostctl` is already in
`sys.modules` - the app, the example test and the package suite all import it -
so every table is registered whatever `env.py` says. Proven by deleting
`env.py`'s import: the full suite stayed at 1124 passed, exit 0, while the test
alone went red. A `python -c` child given exactly `_env_imports()`'s names sees
only what `env.py` names, so the same mutation now fails the canonical gate.
The cost is one subprocess per run; the alternative - reloading the world with
`importlib.reload` - cannot undo a registration on shared metadata.

## Alternatives considered

**`scufris_core/enums.py` for `RunPhase`** - a module holding one enum, an
allowlist entry, and an invitation to grow into the app's enum dumping ground.

**A `RunPhase` re-export from `scufris/enums.py`** - two call sites, and a
permanent second answer to "where does this live".

**Moving `host_watch.py` and `host_approval_bridge.py` into `hostctl`** -
`host_watch` imports eleven root modules (checks, digest, scheduler, the agent
store), most of which v0.2.0 deletes, and `host_approval_bridge` couples an
approval to a conversation that does not exist yet. Either move would make
`hostctl` depend on agents and projects, which is the graph edge the carve is
for. The epic's open question - whether host approvals are conversation events -
decides where the bridge finally lands.

## Consequences

- `core` gains `pydantic`; see the amendment in `tasks/20260803-213242/DECISION.md`.
- A reader of `RunPhase` takes one hop into another distribution, and
  `scufris/enums.py` is now purely the app's option enums.
- A future package's tests must pick a basename no other suite uses, and must
  not add a `conftest.py`. Both failures are loud and immediate collection
  errors, so neither needs a guard.
- `examples/hostctl_approval_flow.py` is the client's runnable contract, and it
  runs in the suite through `tests/test_examples.py`'s `OFFLINE` list.
