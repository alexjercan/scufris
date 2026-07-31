# Split the host, hostd, and auth modules under the size cap

- STATUS: CLOSED
- PRIORITY: 80
- TAGS: refactor, v0.2.0, host, security, backend, maintainability
- KIND: TASK
- FLOW STEP: DONE
- PLAN STATUS: APPROVED
- PARENT: 20260731-171411
- DEPENDS ON: 20260731-171420

## Story

As a maintainer, I want the host, hostd, and auth modules under the size cap,
so that privileged-path changes are reviewable without loading the entire host
stack.

## Steps

- [x] Characterize behavior with the existing host, hostd, and auth suites
      before moving code.
- [x] Split `scufris/hostd/actions.py` (774) by verb family, keeping the
      protocol and dispatch surface in one module.
- [x] Split `scufris/hostconfig.py` (664) by parse/render versus apply.
- [x] Split `scufris/mcp_host_tools.py` (630) by host domain (stats, network,
      thermal, packages, generations).
- [x] Trim or split `scufris/auth.py` (608). The deny-by-default middleware and
      the public-path list stay in one module; only genuinely separable pieces
      move.
- [x] Apply the epic comment policy to every file touched.
- [x] Remove the corresponding allowlist entries from the size guard.

## Definition of Done

- No file under `scufris/host*/`, `scufris/hostd/`, or `scufris/auth.py`
  exceeds 600 lines (cmd: `python scripts/check_file_size.py`).
- One deny-by-default middleware remains, public paths stay only in
  `scufris/auth.py`, and no route declares its own auth dependency
  (cmd: `rg -n "Depends" scufris/`).
- Host verbs, previews, approvals, audit, and inspection unchanged
  (cmd: `python -m pytest tests/test_host_actions.py tests/test_host_action_api.py tests/test_host_inspection.py tests/test_auth.py`).
- Privileged VM tests pass (cmd: `nix build .#scufris-hostd-vm-test`).
- `scufris/hostd/README.md` and `scufris/host/README.md` match the new layout
  (cmd: `rg -n "actions|hostconfig" scufris/hostd/README.md scufris/host/README.md`).

## Notes

- Epic: 20260731-171411.
- Depends on: 20260731-171420.
- Privileged path. Any doubt about a split resolves toward leaving the security
  boundary intact, even if a file stays near the cap. Record the exception in
  the allowlist with a reason rather than weakening the boundary.

## Close-out

**What / why.** Four modules (606 + 664 + 772 + 629) became four packages of the
same name, each with a facade `__init__.py` - the shape 20260731-171428 and
20260731-171429 established. Import paths did not move: NO `tests/` or
`examples/` import line changed, and the only lines touched outside `scufris/`
are seven monkeypatch target strings. Landed as four commits, one per module,
in dependency order (hostd/actions, hostconfig, mcp_host_tools, auth). Each
commit deleted its own `ALLOWLIST` entry, so the guard was green at every
commit rather than only at the tip - proved by
`git rebase master --exec 'python scripts/check_file_size.py'`, which executed
at all four.

| Package | Modules (lines) | Largest |
|-|-|-|
| `scufris/hostd/actions/` | taxonomy 67, models 129, validate 235, plans 361, `__init__` 114 | 361 |
| `scufris/hostconfig/` | models 133, resolve 228, changes 231, render 55, `__init__` 82 | 231 |
| `scufris/mcp_host_tools/` | inspection 354, actions 193, `__init__` 136 | 354 |
| `scufris/auth/` | policy 224, credentials 128, store 247, `__init__` 86 | 247 |

**The Fog question, decided with a measurement.** `scufris/auth.py` is SPLIT,
not trimmed. It was 606 against a 600 cap. The epic's comment sweep found five
lore sites in it, and every one is a citation appended to a sentence that keeps
its invariant - worth at most four lines, landing at 602, still over. Closing
the remaining gap meant cutting the 20-line module docstring, which states the
mechanism and the deny-by-default contract and which the comment policy says to
keep. A file that clears the cap by one line is also one edit from failing the
gate again, which is the ratchet failing at its purpose.

The security boundary is unchanged and was never in this file: the enforcement
point is ONE middleware in `scufris/app.py`, which this task did not touch.
`policy.py` is now the single module answering every question that middleware
asks, `PUBLIC_PATHS` and `PUBLIC_STATIC_PATHS` have exactly one definition site
each, and no route declares its own auth dependency (`rg -n "Depends" scufris/`
has no hits at all).

**Plan corrections, all made while reading the code.**

| Plan said | Landed | Why |
|-|-|-|
| `taxonomy.py` holds `PROTECTED_GENERATIONS`, `SYSTEM_PROFILE`, `SWITCH_UNIT`, the timeouts | `plans.py` | 20260731-171429's rule: assign a constant to the module that READS it. `plans.py` is the only reader of every one |
| `taxonomy.py` holds `PATH_INFO_TIMEOUT` | `validate.py` | same rule; `validate_toplevel` is its only reader |
| `_validate_provenance` moves verbatim | renamed `validate_provenance` | it crossed a module boundary, so its caller is now outside the module that defines it |
| `mcp_host_tools/__init__.py` "keeps INSPECTION, ACTIONS, register" | also binds all 20 public tools by name | needed for `dir(server)` at `test_host_mcp_server.py:336`, as the plan's clause (b) required |

**Two module-level `logger`s dropped.** `hostconfig.py` and
`mcp_host_tools.py` each assigned `logger = logging.getLogger(__name__)` and
never called it. Carrying a dead global into a package means picking one
submodule for it arbitrarily; `rg 'hostconfig.*logger'` and `rg 'logger\.'`
confirmed no reader anywhere. `auth.py`'s logger IS used, in two places that
landed in different submodules, so `credentials.py` and `store.py` each got
their own.

Five other `hostconfig` names stopped being reachable as `scufris.hostconfig.X`
along with it: `MAX_CHANGES`, `MAX_LOG_TAIL`, `GIT_TIMEOUT`, `EVAL_TIMEOUT` and
the `Propose` type alias. Each moved to the submodule that reads it and none is
in the facade, because nothing outside the package consumes any of them -
`Propose`'s only use is inside `changes.py` itself. Deliberate under the epic's
"export only names with a real consumer" rule, recorded here because it is a
public-surface change rather than a pure move.

**Difficulties.**

The plan predicted the `scufris.mcp_host_tools._inspector` patch targets would
fail SILENTLY - the 20260731-171428 class, where the old target still resolves
through the facade while the code reads the submodule's own global. Measured
rather than assumed: setting one back to the old string and running the suite
gives **4 failed, 21 passed** with an `AttributeError`, because the facade does
not bind `_inspector` at all, so `monkeypatch.setattr` has nothing to replace.
The repoint was still required; the failure mode was the LOUD one. Same for the
three `scufris.auth.time.time` targets: reverting them gives 2 failed with an
`ImportError`. Both sabotage checks were run and reverted.

`ruff format --check tests/` flags `tests/test_host_mcp_server.py`. Measured
against the PRE-move file before writing this down, per the epic's rule:
`git show master:tests/test_host_mcp_server.py` extracted to a scratch path and
checked there reports "Would reformat" on the base too. Inherited, not
introduced - ten `tests/` files are in that state on master, including
`test_supervisor.py` and `test_mcp_server.py`, which this task never touched.
The canonical gate runs `ruff check .` (the linter), not `ruff format`, so this
is outside the gate either way and is not this task's to fix.

No class was cut in any of the four splits - the largest classes are
`ConfigChangeBuilder` (126), `SessionStore` (141) and `LoginThrottle` (87), all
well inside the cap - so 20260731-171429's arithmetic problem never arose, and
neither did its nine-attribute-reach surprise: `rg 'obj\._'` over the test tree
had nothing to report because no member moved off an object.

**Evidence.**

| Proof | Result |
|-|-|
| `python scripts/check_file_size.py` | green; 4 allowlist hits on base, 0 after |
| `git rebase master --exec 'python scripts/check_file_size.py'` | executed and passed at all 4 commits |
| `git diff master -- scripts/check_file_size.py` | exactly the 4 entries this task owns removed; `scufris/app.py`, the 7 `tests/` and 5 `web/src/` entries untouched |
| `python -m pytest -o addopts=""` | 896 passed, same as the recorded baseline |
| targeted 7-file host/auth run | 220 passed |
| `git diff master -- tests/ examples/` | 2 files, 7 changed lines, every one a monkeypatch target string; no import line changed |
| `rg '^PUBLIC_PATHS\|^PUBLIC_STATIC_PATHS' scufris/` | 2 hits, both `scufris/auth/policy.py` |
| `rg -n 'Depends' scufris/` | no hits |
| `rg 'scufris\.mcp_host_tools\.inspection\._inspector' tests/` | 4 |
| lore sweep over the four trees | no hits (9 on base) |
| `rg '[0-9]{8}-[0-9]{6}' scufris/ -g '!*.md'` | no hits |
| `nix build .#scufris-hostd-vm-test` | built |
| `nix flake check` | **all checks passed** - 7 checks including `records`; the known tatr-0.1.0 false positive did not fire |

Ten lore sites swept, each keeping its invariant as a fact about the code:
`auth.py` 53 (`see DECISION.md`), 261, 378, 471, 565 and the `DECISION.md`
deployment-boundary citation in `LoginThrottle`; `hostd/actions.py` 79 (`R1.5`)
and 745 (`R1.4`); `mcp_host_tools.py` 442 (`R1.6`) and 461 (`R1.11`). The
`R0`-`R4` risk-class names in `hostd/actions/` are the domain taxonomy and were
kept, but `RiskClass`'s docstring lost "the spike's" - a spike reference as
its only justification. Three READMEs updated: `scufris/README.md` (diagram, two module-map rows,
the R3 endpoint row), `scufris/hostd/README.md` (three sites),
`scufris/host/README.md` (two sites).

**Reflection.** The plan's most valuable clause was the one it inherited rather
than derived: "when a check fires on MOVED code, run it against the pre-move
file before writing down a cause." It fired twice this task and the answer was
"inherited" both times, which is only worth saying because it was MEASURED both
times - and the measurement was nearly botched once, when a `git checkout
master` inside the worktree failed (the branch is checked out elsewhere) and
silently compared the branch to itself. `git show master:<path>` into a scratch
file is the form that works from a worktree; the checkout form is not.

The second thing worth carrying: the plan predicted a silent failure mode and
got a loud one. Predicting WHICH way a patch-target break fails is not
reliable - it depends on whether the facade happens to bind the name - so the
useful discipline is not the prediction but the sabotage check that settles it
in one run. Both were cheap and both are now recorded facts rather than
reasoning.
