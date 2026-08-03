# Decision: The host carve routes every root import through a widened facade

- DATE: 20260803-215500
- STATUS: ACCEPTED
- TASK: 20260803-214748
- TAGS: architecture,packaging,host

## Context

`scufris/` imports the host tree's INTERNALS in thirteen places:
`hostd/engine.py:33`, `hostd/preview.py:26-29`, `hostd/nixos.py:41-43`,
`hostd/executor.py:25`, `hostd/actions/plans.py:12-13`,
`hostd/actions/validate.py:18`, `hostconfig/resolve.py:17`,
`hostconfig/changes.py:23`, `app.py:57` and `api/host.py:33`. They name
`host.run`, `host.models`, `host.storage`, `host.units` and `host.overview`.

Today those are relative intra-distribution imports and nothing objects. The
moment the tree becomes `scufris_host`, every one reads
`from scufris_host.<internal> import ...` - the one thing the epic's single rule
forbids. `test_no_package_imports_a_sibling_private_module` exists for exactly
this and is green today only because `core` gave it nothing to police; its own
docstring says it "earns a red run once a second package is carved out beside
`core`". This carve is that run, so the rule stops being decorative here or it
never starts.

A second fact shapes the same choice. Inspection confirms the whole moving tree
- `scufris/host/*.py`, `metrics.py`, `processes.py` - imports nothing but
stdlib, `psutil` and `pydantic`. No `scufris` module, not even `logsetup`; no
database. The epic's `host -> nothing` edge is literally true, so the package
declares no `scufris-core` dependency.

## Decision

**1. Widen `scufris_host/__init__.py`; rewrite the thirteen call sites.** The
existing `__all__` already exports every name the thirteen call sites want but
three. Add `nix_cli` (from `run`), `MIN_HOST_OVERVIEW_TTL` and
`HostOverviewCache` (from `overview`), plus `NIX_FEATURES` (from `run`) for the
root test and example that build a `FakeRunner` key from it, and re-export
`metrics`' and `processes`' public names. Then no root module reaches past the
facade.

**2. `metrics.py` and `processes.py` move into the package, flat.** Both
qualify by the same inspection, and `HostStats` is what Stats serves. Their
public names join the flat facade rather than staying addressable as
`scufris_host.metrics`. Verified the three name sets are pairwise disjoint -
`host.__all__`'s sixty-three, metrics' sixteen names, processes' six - so the
flat facade needs no aliasing.

**3. `HostInspector`/`HostOverview` move to `inspector.py`; `__init__` becomes
purely re-exports.** Discovered during the work, and it is what makes decision 1
possible at all. `overview.py` imports `HostInspector` and `HostOverview` from
the package root, and both are defined below `__init__`'s import block, so
adding `from .overview import ...` to the facade is an import cycle that fails
at runtime. Moving the two classes to their own module removes the cycle
outright rather than working around it, and leaves the distribution root as a
door and nothing else - which is the shape `hostd` and `hostctl` inherit when
they carve.

## Alternatives considered

**Exempt the root distribution from the rule.** Rejected. It exempts the tree
the epic itself names as "the one most likely to keep an import pointing at
where a module used to sit", which retires the check on its first real use and
leaves the boundary as a README paragraph again - the state
`tests/test_package_boundaries.py` was written to end.

**Leave `run.py` and `models.py` at the root, move only the report modules.**
Rejected. It splits `host.run` from the reports it serves, leaves
`Runner`/`CommandResult` straddling two distributions, and hands `hostd`'s later
carve the same problem in a worse shape - the epic already argues that
duplicating those types across the wire protocol, or hoisting a module that
shells to `nix` into `core`, are both worse than the trio's real edges.

**Keep `metrics`/`processes` as addressable submodules.** Rejected. `telegram`,
`api` and `app` all import them, so it re-creates decision 1's problem
immediately for no benefit; the disjoint name sets mean flat costs nothing.

**Break the `overview` cycle with a `TYPE_CHECKING` guard instead of moving the
classes (decision 3).** Rejected. It works - `overview.py` has
`from __future__ import annotations` and every use of the two names is an
annotation - but it leaves a load-bearing constraint that is invisible at the
use site: the first runtime use of `HostInspector` added to that module
reintroduces the cycle, with a `NameError` rather than an import error to
diagnose it by. The alternative of keeping the import at the BOTTOM of
`__init__.py` was rejected for the same reason plus a `# noqa: E402` whose
justification is a cycle.

## Consequences

- Four names added to `__all__` plus twenty-two re-exports; thirteen import
  lines rewritten. Small, because the facade was already nearly complete.
  Derived, not recalled: `__all__` goes from 63 entries on `master` to 89, and
  the `from .metrics import` and `from .processes import` blocks carry 16 and 6
  names.
- `psutil` leaves the root `dependencies`: after the move zero root modules
  import it. `types-psutil` stays in the dev group.
- `scufris_host` is a flat namespace. A future name added to `metrics` or
  `processes` must not collide with a report name; the facade's single `__all__`
  makes a collision a visible edit rather than a silent shadow.
- `hostd` and `hostctl`, carved later, inherit a facade that already exports
  what they use, so their carves do not reopen this.
