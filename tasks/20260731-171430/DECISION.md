# Decision: four independent package splits, and auth is split rather than trimmed

- STATUS: ACCEPTED
- DATE: 2026-07-31
- TASK: 20260731-171430
- TAGS: refactor, maintainability, kiss, host, security
- EPIC: 20260731-171411

## Context

Four modules are over the 600-line source cap and allowlisted in
`scripts/check_file_size.py`. Measured on the base branch:

| Module | Lines | Over by |
|-|-|-|
| `scufris/auth.py` | 606 | 6 |
| `scufris/hostconfig.py` | 664 | 64 |
| `scufris/hostd/actions.py` | 772 | 172 |
| `scufris/mcp_host_tools.py` | 629 | 29 |

Sibling tasks 20260731-171428 and 20260731-171429 settled the epic's shape: an
oversized module becomes a PACKAGE of the same name with a facade `__init__.py`,
import paths do not move so no call site changes, and submodules import each
other directly, never through their own `__init__`. 171429 added the sizing rule
the shape does not cover: when most of a module is one class, measure the method
groups and do the arithmetic before choosing the cut.

That second rule does not bind here. None of the four modules is dominated by a
single class - the largest classes are `ConfigChangeBuilder` (126),
`SessionStore` (141) and `LoginThrottle` (87), all comfortably under the cap.
Every cut below is a module-level grouping, so no class is cut at all.

## Decision

### 1. All four become packages of the same name

The shape carries over unchanged. Measured group sizes, and the resulting
layouts:

**`scufris/hostd/actions/`** (772 lines; a nested package inside `scufris/hostd/`)

| Module | Owns | ~Lines |
|-|-|-|
| `taxonomy.py` | `RiskClass`, `ActionKind`, `RISK_OF`, `UNIT_KINDS`, `R3_KINDS`, `_SYSTEMCTL_VERB`, `ActionRefused`, the profile paths and timeouts | 115 |
| `models.py` | the five arg models, `ActionArgs`, `_ARGS_MODEL`, `Step`, `Plan`, `parse_args` | 137 |
| `validate.py` | the unit-name regexes, suffix tuples, `DENIED_UNIT_STEMS`, `_STORE_PATH`, `_REVISION`, `normalise_unit`, `validate_toplevel`, `_validate_provenance` | 245 |
| `plans.py` | `_switch_step`, the generation helpers, `_activate_plan`, `_rollback_plan`, `build_plan` | 325 |
| `__init__.py` | facade | 60 |

**`scufris/hostconfig/`** (664 lines)

| Module | Owns | ~Lines |
|-|-|-|
| `models.py` | `ConfigChangeRefused`, `ChangeState`, `Resolved`, `ConfigChange`, the three build events, `_build_error_*` | 130 |
| `resolve.py` | `default_attr`, `_git`, `_validate_ref`, `resolve`, `flake_url`, `build_argv`, `check_attr`, `toplevel_from` and their regexes and timeouts | 240 |
| `changes.py` | `UnknownChange`, `ChangeInFlight`, `ConfigChangeStore`, `config_supervisor`, `ConfigChangeBuilder` | 250 |
| `render.py` | `render_change` | 90 |
| `__init__.py` | facade | 55 |

**`scufris/mcp_host_tools/`** (629 lines)

| Module | Owns | ~Lines |
|-|-|-|
| `inspection.py` | the collectors, `_human_bytes`, `_format_processes`, `_inspector`, `_scope`, `_bad_scope` and every read-only tool | 383 |
| `actions.py` | `propose_host_action`, `host_action_status`, `propose_nixos_change`, `nixos_change_status`, `host_action_audit` and the two renderers | 206 |
| `__init__.py` | facade, `INSPECTION`, `ACTIONS`, `register` | 100 |

**`scufris/auth/`** (606 lines)

| Module | Owns | ~Lines |
|-|-|-|
| `policy.py` | cookie and header names, `API_TOKEN_ENV`, `PUBLIC_PATHS`, `PUBLIC_STATIC_PATHS`, `UNSAFE_METHODS`, `OPERATOR_ONLY_PATTERN`, `operator_only`, loopback, `AuthConfigError`, `auth_required`, `validate_auth_config`, `same_origin`, `safe_next_path`, `session_cookie_kwargs` | 243 |
| `credentials.py` | scrypt parameters, `hash_password`, `verify_password`, `mint_api_token`, `token_matches`, `bearer_token` | 115 |
| `store.py` | `Session`, `SessionStore`, `LoginThrottle`, `now` | 257 |
| `__init__.py` | facade | 65 |

### 2. `scufris/auth.py` is SPLIT, not trimmed

The epic's Fog left this open. The measurement settles it: `auth.py` is 606,
six over. The comment sweep this epic mandates finds five lore sites in the file
(`see DECISION.md` at 53, `Review round 1` at 261, 378, 471 and 565) and every
one is a citation appended to a sentence that keeps its invariant. The sweep is
worth at most four lines. 606 - 4 = 602, still over the cap.

Reaching 600 by trimming requires cutting the 20-line module docstring, which
states the mechanism and the deny-by-default contract and which the epic's own
comment policy says to keep. And a file that clears the cap by one line is one
edit from failing the gate again, which is the ratchet failing at its purpose.

So `auth.py` splits. The security constraint is met by the cut, not despite it:
the enforcement point is not in this file at all - it is ONE middleware in
`scufris/app.py` - and `policy.py` is the single module that answers every
question that middleware asks (is this path public, is this method unsafe, is
this an operator-only path, is authentication required at all). `PUBLIC_PATHS`
and `PUBLIC_STATIC_PATHS` keep exactly one definition site.

### 3. Four commits, one per module

Unlike 20260731-171429, these are four INDEPENDENT modules with no shared
seam, so the 171428 pattern applies: one commit per split boundary, each
deleting its own `ALLOWLIST` entry, so the guard is green at every commit rather
than only at the tip. Order: `hostd/actions`, `hostconfig`, `mcp_host_tools`,
`auth` - deepest dependency first.

## Rationale

- Every cut is at a group that owns its own reason to exist, not at a size
  boundary. `validate.py` holds the argument-validation invariant this file's
  docstring is built around ("an argument may not become a flag"), which is a
  property to review in one place. `resolve.py` holds the rule that the store
  path is built here from a revision resolved here. `inspection.py` versus
  `actions.py` is the audience split `register` already encodes physically.
- No class is cut, so 171429's mixin trap does not arise.
- The facade keeps every import path, so none of the 61-plus call sites move and
  no `tests/` or `examples/` import line changes.

## Consequences

- Four monkeypatch target strings `scufris.mcp_host_tools._inspector`
  (`tests/test_host_mcp_server.py:211,245,278,313`) must be repointed to
  `scufris.mcp_host_tools.inspection._inspector`. This is the SILENT failure
  mode 20260731-171428 paid for: after the split the old target still RESOLVES
  through the facade while the tools read `inspection`'s own global, so the test
  passes and patches nothing.
- Three targets `scufris.auth.time.time`
  (`tests/test_auth.py:296,314,316`) reach the stdlib `time` module through
  `scufris.auth`. The facade will not bind `time`, so these fail LOUDLY with
  `AttributeError`; repointed to `scufris.auth.store.time.time`.
- `tests/test_host_mcp_server.py:336` does `import scufris.mcp_host_tools as
  server` and enumerates `dir(server)` for public callables to assert no
  approving tool exists. The facade must bind every public tool by name or the
  enumerated set silently shrinks and the assertion becomes vacuous.
- Task records and LESSONS.md entries citing `auth.py:<line>`,
  `hostconfig.py:<line>`, `actions.py:<line>` or `mcp_host_tools.py:<line>` no
  longer resolve. They are history and are not rewritten.

## Alternatives considered

- **Trim `auth.py` instead of splitting.** Rejected on the measurement above:
  four lines available, six needed, and the only remaining source is contract
  the comment policy keeps.
- **Split `auth.py` once (`store.py`) and leave the other 374 lines in
  `__init__.py`.** Clears the cap with the minimum cut, but puts real code in
  the facade, which every other package in this epic keeps free of it. Rejected
  for consistency at the cost of one extra file.
- **Split `mcp_host_tools.py` by host domain (stats, network, thermal,
  packages, generations), as the task's Step suggested.** Rejected: five
  modules of 40-80 lines each, none of which owns anything the others do not,
  and the audience split - which `register` already encodes - is the seam that
  carries meaning. Two submodules is the KISS answer.
- **Keep `scufris/hostd/actions.py` flat and move only `plans.py` out.**
  772 - 299 = 473, which clears the cap. Rejected: it leaves the taxonomy, the
  models and the argv validation interleaved in one file, and the validation
  invariant is the property most worth reviewing alone.
- **Split `hostconfig.py` into `store.py` and `builder.py` separately.**
  Rejected: `ConfigChangeStore` (70) has one reader, `ConfigChangeBuilder`, and
  the merged `changes.py` is 250 - well inside the cap. One caller is not an
  abstraction.
