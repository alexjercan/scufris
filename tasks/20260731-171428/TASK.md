# Split the agent runtime modules under the size cap

- PRIORITY: 90
- TAGS: refactor, v0.2.0, agents, backend, maintainability
- KIND: TASK
- ACTIVITY: COMPOUNDING
- GATES: PLAN REVIEW RETRO
- RESOLUTION: DONE
- PARENT: 20260731-171411
- DEPENDS ON: 20260731-171420

## Story

As a maintainer, I want the agent runtime split along its real seams, so that
changing one backend, one store, or the session index does not require loading
four thousand lines of unrelated agent code.

## Steps

Each module becomes a package of the same name (DECISION.md); import paths do
not move, so no caller changes. Submodules import each other directly, never
through their own `__init__`. One commit per split, in this order (the import
direction is `sessions -> agent -> backends`; `agent_store` is independent).

- [x] Record the baseline: `python -m pytest` green and its test count, plus
      `python scripts/check_file_size.py` green on the four allowlisted files.
      Ruff/mypy must be green before any move, so a later failure is this
      change's.
- [x] Split `scufris/sessions.py` (835) into `scufris/sessions/`:
      `steering.py` (the `[scufris-tools]` preambles, `_STEER_RE`,
      `strip_steering`), `models.py` (`RateWindow`, `UsageQuota`, `ToolCall`,
      `TokenUsage`, `SessionInfo`, `SessionContext`, `TranscriptMessage`),
      `rollout.py` (`resolve_codex_home`, rollout discovery and event iteration,
      `list_sessions`, `rollout_mtime`, `delete_session`, `read_context`),
      `transcript.py` (`read_transcript`, `reasoning_fingerprint`,
      `merge_reasoning`, `format_fork_seed`), `usage.py` (`read_usage`,
      `MemoryFootprint`, `read_memory_footprint`), and an `__init__.py` facade.
      `models.py` must keep importing nothing from `scufris` beyond `config`, so
      `agent` and `backends` can depend on it without a cycle.
- [x] Split `scufris/agent.py` (832) into `scufris/agent/`: `events.py`
      (`AgentUnavailable`, `AgentReply`, the `Stream*` models and the
      `StreamEvent` union, `STREAM_READ_LIMIT`, the `ToolCall`/`TokenUsage`
      re-export), `env.py` (`_resolve_codex_bin`, `agent_subprocess_env`,
      `_codex_env`, `login`), `mcp.py` (`ScufrisMcpServer`,
      `scufris_mcp_servers`, `_server_override`, `_mcp_overrides`),
      `appserver.py` (`_steer`, `_turn_mode`, the app-server event/usage
      parsers, `_git_writable_roots`, `_sandbox_overrides`, `_appserver_call`,
      `_stream_app_server`), and an `__init__.py` facade. Decide once
      `appserver.py` exists: at or under 600 -> keep it; over -> move
      `_git_writable_roots` and `_sandbox_overrides` to `sandbox.py`.
- [x] Split `scufris/backends.py` (1098) into `scufris/backends/`: `base.py`
      (`BackendStatus`, `_context_from_status`, the `AgentBackend` protocol,
      `_LAST_MESSAGE_PREVIEW`), one module per adapter - `codex.py`,
      `claude.py` (its stream/transcript parsers and arg builders included),
      `opencode.py`, `mock.py` - each owning its own permission/sandbox/tool
      mode map, and an `__init__.py` holding `get_backend` and `session_info`.
      The protocol stays exactly one module; `get_backend` stays the only place
      that knows every adapter.
- [x] Split `scufris/agent_store.py` (1032) into `scufris/agent_store/`:
      `records.py` (the four error types, `AGENT_ID_RE`, `AgentRecord`,
      `AgentLifecycle`, `SessionIdList`, `_slugify`, `RESERVED_AGENT_IDS`),
      `registry.py` (`SessionRegistry`), `outcomes.py` (`RunOutcome`,
      `OutcomeStore`), `reserved.py` (the settings-derived synthetic
      orchestrator and host records), `store.py` (`AgentStore`), and an
      `__init__.py` facade. Decide once `store.py` exists: at or under 600 ->
      keep it; over -> move the signal writers (`awaiting_approval`,
      `request_input`, `report_back`) and the outcome queries onto
      `OutcomeStore`, with `AgentStore` keeping the existence guard and
      delegating.
- [x] Apply the epic comment policy to every file touched as it moves: delete
      the phase-code lore (`A0`, `A3`, `B5b`, `BC1`, `R1.1`, ...) that only
      cites a record, keep every invariant as a fact about the code, and
      introduce no task IDs.
- [x] Delete `scufris/agent.py`, `scufris/agent_store.py`, `scufris/backends.py`
      and `scufris/sessions.py` from the guard's `ALLOWLIST` in the same commit
      that lands each split.
- [x] Update the `scufris/README.md` module map rows for the four packages.
- [x] `git add` the new packages BEFORE running `nix flake check`: it evaluates
      only tracked files.

## Definition of Done

- The four packages are all at or under 600 lines per file and the allowlist no
  longer names them - 4 hits on base, none after
  (cmd: `rg -n "scufris/(agent|agent_store|backends|sessions)\.py" scripts/check_file_size.py; python scripts/check_file_size.py`).
- No import contract drifts: the suite passes with the same count as the
  recorded baseline and no `tests/` import line changed
  (cmd: `python -m pytest`; `git diff --stat -- tests/`).
- Backend selection and health remain identical for codex, claude, opencode,
  and mock (test: `test_backends.py`).
- Session ownership and multi-session history unchanged
  (test: `test_sessions.py`).
- Agent run lifecycle, resume, cancel, and outcomes unchanged
  (test: `test_agent_store.py`, `test_agent.py`).
- No task ID enters code, Markdown excluded
  (cmd: `rg -n "[0-9]{8}-[0-9]{6}" scufris/ -g '!*.md'`).
- `scufris/README.md` module map matches the new layout
  (cmd: `rg -n "backends|agent_store|sessions" scufris/README.md`).
- The full backend gate passes, `records` excepted while this task is open
  (cmd: `nix flake check`).

## Notes

- Epic: 20260731-171411. Depends on 20260731-171420 (guard and comment policy).
- Package-facade shape and the rejected alternatives: DECISION.md.
- The guard's `ALLOWLIST` ratchets both ways: an entry left behind after its
  file is gone or back inside the cap fails the gate.
- `nix flake check` evaluates only git-TRACKED files; untracked new modules fail
  it with a `/build/work/...` path.
- `flake.nix` pins tatr 0.1.0, which predates the v2 record schema, so the
  `records` check reports `unplanned-in-progress` while this task is
  IN_PROGRESS. Known false positive; do not chase it and do not bump the pin.
- Split along ownership, not line count. A mechanical split that keeps the same
  coupling is a failure.
- Do not fold in behavior fixes; file a task instead.
- Assumption: no `CHANGELOG.md` entry. The change is internal with no observable
  behavior, setting, or interface change.
- `scufris/auth.py` (606), `scufris/hostconfig.py` (664) and
  `scufris/mcp_host_tools.py` (629) are over the cap but belong to sibling
  tasks; leave their allowlist entries alone.

## Close-out

**What / why.** Four modules (835 + 832 + 1098 + 1032 lines) became four
packages of the same name, each with a facade `__init__.py` - the shape
`scufris/host/` and `scufris/hostd/` already use. Import paths are unchanged, so
none of the 61 call sites moved and no `tests/` import line changed. Submodules
import each other directly (`from .models import ToolCall`), never through their
own package `__init__`. Landed as one commit per split, in dependency order:
sessions, agent, backends, agent_store. Each commit deleted its own `ALLOWLIST`
entry, so the guard was green at every commit rather than only at the end.

**Both decide-once points resolved.**

| Decision | Measured | Outcome |
|-|-|-|
| `agent/appserver.py` gets a `sandbox.py`? | 466 lines | No. Kept the smaller layout. |
| `agent_store/store.py` sheds its signal writers? | 667 lines with only `reserved` split off | Yes. `awaiting_approval`, `request_input`, `report_back` and the outcome queries moved onto `OutcomeStore`; `AgentStore` keeps `_require_exists` and delegates. `store.py` is now 595. |

The second one was not a size trick: the signal writers only ever built a
`RunOutcome` and handed it to `self._outcomes.set`, so they were `OutcomeStore`
methods living on the wrong class. `AgentStore` retains exactly the part that
needs it - the existence guard over `self._agents`.

**Alternatives.** None reopened; the shape and the rejected alternatives are in
DECISION.md and were not relitigated.

**Difficulties.** One genuine conflict with "`tests/` byte-identical": three
monkeypatch TARGET STRINGS bound to a module's globals and could not survive the
move - `scufris.agent.STREAM_READ_LIMIT`, `scufris.agent.shutil.which` and
`scufris.backends._stream_app_server`. Patching the package attribute would no
longer reach the submodule global the code actually reads, so the tests would
have passed while patching nothing. Repointed 5 strings across `test_agent.py`
and `test_backends.py` (`...agent.appserver.STREAM_READ_LIMIT`,
`...agent.env.shutil.which`, `...backends.codex._stream_app_server`). These are
patch targets, not import lines, so the Definition of Done clause holds as
written.

`ruff format --check` flagged the new package. The cause was the delegating
calls written during the split, not anything inherited: all four pre-split
modules are format-clean at the merge-base (`ruff format --check` on them reports
"4 files already formatted"). Ran `ruff format` on the new package.

**Evidence.**

| Proof | Result |
|-|-|
| `python scripts/check_file_size.py` | green; 4 allowlist hits on base, 0 after |
| `python -m pytest` | 896 passed, same as the recorded baseline |
| `git diff --stat -- tests/` | 2 files, 5 lines - the monkeypatch targets above |
| `rg -n "[0-9]{8}-[0-9]{6}" scufris/ -g '!*.md'` | no hits |
| `nix flake check` | ruff, mypy, pytest, filesize green; `records` reports the known tatr-0.1.0 false positive |

Largest file in the four packages is now 595 lines; the phase-code lore
(`A0`, `A3`, `B5b`, `BC1`, `R1.1`, "part 1/3", "review round 1") is gone from
every docstring that moved, kept as facts about the code where it was an
invariant. One deliberate STRING edit rides along, outside that comment sweep:
`_ORCHESTRATOR_DESCRIPTION` loses `(from B5c)`. It is user-visible text rather
than a comment, so it is called out here on its own; nothing asserts on it.

**Reflection.** Measuring before deciding was worth the extra step twice: the
plan's guess that `appserver.py` would need a `sandbox.py` was wrong, and the
guess that `store.py` might not need the second cut was also wrong. Next time,
grep the string monkeypatch targets alongside the import sites at survey time -
they are call sites that no import-path grep reports, and they were the only
thing in this refactor that could have failed silently.
