# Decision: what replaces the same-wheel guarantee when hostd becomes its own distribution

- DATE: 20260804-020000
- STATUS: ACCEPTED
- TASK: 20260803-214747
- TAGS: v0.2.0, architecture, packaging, host

## Context

`scufris/hostd/` and the app ship from ONE wheel today, and the root
`pyproject.toml` says so in a comment: the two halves of a socket protocol
cannot drift in version because there is only one artifact. Carving the helper
into `packages/hostd` destroys that guarantee by construction, so the carve is
only honest if something else takes it over.

Four further choices fall out of the same move: whether the HTTP error mapper may
keep importing the helper's wire codes, whether root importers go through the
facade, what happens to a test whose AST sweep is rooted at `scufris/`, and
whether `hostd -> core` is a real graph edge. NOTES.md left all of them open;
the parent epic named two of them as this task's to answer.

## Decision

**1. The drift guard is a file-based pin check.**
`test_the_app_pins_hostd_to_one_exact_version` in `tests/test_release.py` reads
both `pyproject.toml` files with `tomllib` and asserts the root's
`scufris-hostd` requirement is `==` the version `packages/hostd/pyproject.toml`
declares. The parent DoD's
`test_hostd_and_app_report_the_same_protocol_version` is dropped: there is no
app-side protocol version to compare. `PROTOCOL_VERSION` appears in
`protocol.py` and the facade and nowhere else, and `hostclient` performs no
handshake, so a test of that name would compare a constant with itself or
require inventing an app-side copy - a behavior change this task forbids.

**2. `api/errors.py` keeps the helper's error codes**, re-pointed to
`from scufris_hostd import ErrorCode`.

**3. Every root SOURCE importer goes through the facade `scufris_hostd`**, and
the facade gains exactly one name to allow it: `encode`, from `protocol`.
`hostclient.py`'s three submodule imports collapse into one. The moved tests
keep their submodule imports.

**4. The agent-shell spawn sweep follows the code**: root
`test_no_agent_subprocess_is_spawned_without_the_stripped_environment` at
`packages/*/src/*` as well as `scufris/`, and re-point the `executor.py`
exemption at its new path.

**5. `hostd -> core` is a real edge.** `packages/hostd/pyproject.toml` declares
`scufris-core` alongside `scufris-host`, and the parent epic's graph line is
amended from `hostd -> host`.

## Alternatives considered

**`importlib.metadata.version()` equality (NOTES.md option a).** Rejected on two
counts. Environment-dependent: `importlib.metadata.version("scufris")` raises
`PackageNotFoundError` in the repo's local `.venv` today. And it tests the wrong
property - in a workspace both members are always installed from one tree at one
version, so the comparison is true by construction and stays green with a stale
pin.

**A `hello` handshake in `hostclient` (NOTES.md option b).** Rejected here, not
forever. It is a behavior change on the wire, and it is the right answer only
once the app venv and the running root unit can actually resolve different
builds - which needs evidence from a real deployment split.

**The app declaring its own error codes.** Rejected: two definitions of one wire
vocabulary on two sides of a socket is the failure the carve exists to prevent.
`ErrorCode` is part of `hostd`'s published contract, and a client translating it
into HTTP status is that client doing its job.

**Leaving the spawn sweep on `scufris/`.** Rejected: it would sit exactly on its
`assert checked >= 5` floor (6 sites today, 5 after the move) with a dead
exemption entry, and every remaining carve child moves more spawning code out.

**`hostd/main.py` configuring its own logging** to preserve the `hostd -> host`
diagram. Rejected: 87 duplicated lines to avoid amending one line of a record.

**Submodule imports left as-is (NOTES.md open question 5 treats the facade as
optional).** Not actually available:
`test_no_package_imports_a_sibling_private_module` fails any
`scufris_hostd.<submodule>` import from a source root.

## Consequences

The exact pin binds exactly one consumer - someone installing the published
`scufris` wheel - and nothing in this repository would notice it rotting, which
is precisely why it needs a test. Verified in scratch:

- `[tool.uv.sources] X = { workspace = true }` DROPS the version specifier. A
  root declaring `scufris-hostd==0.2.0` against a member at `0.1.0` resolved and
  synced clean; `uv.lock` recorded only
  `{ name = "scufris-hostd", editable = "packages/hostd" }`.
- The built wheel nonetheless carries `Requires-Dist: scufris-hostd==0.2.0`.
- `nix build` resolves from the lock, so it inherits uv's blind spot.

`scripts/check-release-ready.sh` already forces every `packages/*/pyproject.toml`
to the root version; the new test extends that agreement to the one place the
release script cannot see. The release procedure gains one obligation: bump the
pin with the member versions.

Deployment gains a second flake output. `packages.scufris` stops carrying
`bin/scufris-hostd`, so `nix/scufris-hostd.nix` must default to
`packages.scufris-hostd`, and the parent epic's Done Means item 9 is amended to
build that output instead. This breaks at BUILD time, which is why
`scufris-hostd-vm-test` is the proof.

Nothing an operator sees changes: same unit, same socket, same
`PROTOCOL_VERSION = 1`, same audit lines.
