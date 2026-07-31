# Decision: split each oversized agent runtime module into a package facade

- STATUS: ACCEPTED
- DATE: 2026-07-31
- TASK: 20260731-171428
- TAGS: refactor, maintainability, kiss
- EPIC: 20260731-171411

## Context

`scufris/agent.py` (832), `scufris/agent_store.py` (1032),
`scufris/backends.py` (1098) and `scufris/sessions.py` (835) are over the epic's
600-line source cap. 61 import sites across `scufris/` and `tests/` name these
four module paths, several of them private helpers (`_codex_env`, `_steer`,
`_mcp_overrides`, `_stream_app_server`).

The task forbids compatibility shim modules and requires either stable public
import paths or every caller updated in the same change. Those two rules pull in
opposite directions, so the shape of the split is load-bearing.

## Decision

Each of the four modules becomes a PACKAGE of the same name, with the public
surface declared in `__init__.py` and the implementation in sibling submodules:

| Package | Submodules |
|---------|------------|
| `scufris/sessions/` | `steering`, `models`, `rollout`, `transcript`, `usage` |
| `scufris/agent/` | `events`, `env`, `mcp`, `appserver` |
| `scufris/backends/` | `base`, `codex`, `claude`, `opencode`, `mock` |
| `scufris/agent_store/` | `records`, `registry`, `outcomes`, `reserved`, `store` |

Import paths stay `scufris.sessions`, `scufris.agent`, `scufris.backends`,
`scufris.agent_store`; no caller changes and no `.py` file is left behind
forwarding to a new one.

Submodules import from each other DIRECTLY (`from .models import ToolCall`),
never through their own package `__init__`, so no import runs against a
partially initialized package.

The package layering stays acyclic and unchanged in direction:
`sessions -> agent -> backends`.

## Rationale

- `scufris/host/` and `scufris/hostd/` are already packages with a facade
  `__init__.py` that re-exports its submodules' public names. This is the
  repository's existing pattern for exactly this situation, not a new one.
- A package `__init__` IS the module its callers import. The shim the task bans
  is a leftover `backends.py` that forwards to a new module beside it - a second
  path to the same code. A facade leaves one path.
- Renaming to flat siblings (`backends_claude.py`, ...) would churn 61 import
  sites, including test imports of private helpers, for no gain in the seam.

## Consequences

- Callers and tests are untouched, so a green suite is evidence about the moved
  code rather than about a rewritten import surface.
- Each `__init__.py` carries the public surface of its package; adding a name to
  that surface is now an explicit edit rather than a side effect of defining it.
- Submodules may not import their own package `__init__`, and a future reviewer
  reading a package must read `__init__.py` first to see the seam.
- `scufris/agent.py` and its three siblings disappear as paths: task records and
  LESSONS.md entries that cite `agent.py:583`-style locations no longer resolve.
  Those records are history and are not rewritten.

## Alternatives considered

- **Flat sibling modules plus a caller sweep.** Rejected: the churn is the whole
  cost, it makes the diff unreviewable against the behavior-preserving claim,
  and it buys nothing the package does not already give.
- **Keep the modules and only shave comments.** Rejected: the four files are
  1.5x-1.8x the cap; no comment policy closes that gap, and the epic asks for
  the seam, not for shorter files.
- **Split `AgentStore` into a base class plus a subclass.** Rejected as a
  mechanical split: inheritance keeps every line of the coupling it claims to
  break. The record/registry/outcome/reserved seams are real ownership
  boundaries and get the class under the cap on their own.
