# Notes: Move the root helper into packages/hostd

Goal in one line: lift `scufris/hostd/` out into its own distribution
`scufris-hostd`, keeping the socket protocol, the verbs and the audit byte-for-byte
identical - and replace the "same wheel" drift guarantee with a test, because two
distributions can no longer give it by construction.

**This task cannot be done first as written.** `hostd` is not import-clean: it
depends on `scufris.host` and `scufris.logsetup`. See open question 1, which is
blocking.

## What changes

Nothing an operator sees. `scufris-hostd` starts from the same systemd unit, on
the same socket, speaking `PROTOCOL_VERSION = 1`, writing the same audit lines.

What a MAINTAINER sees:

| Before | After |
|---|---|
| `scufris/hostd/` inside the `scufris` distribution | `packages/hostd`, distribution `scufris-hostd`, import root `scufris_hostd` |
| both console scripts ship from one wheel | two distributions, `scufris` pinned to `scufris-hostd == <version>` |
| version drift impossible by construction | `test_hostd_and_app_report_the_same_protocol_version` |
| `packages.scufris` carries `bin/scufris-hostd` | it does NOT - see open question 2 |

## Surfaces

Moves to `packages/hostd/src/scufris_hostd/` (2400 lines, 14 modules):

| File | Lines | Why |
|---|---|---|
| `hostd/engine.py` | 587 | proposals, the four apply refusals, audit calls |
| `hostd/preview.py` | 525 | what the operator sees, and the honesty label |
| `hostd/nixos.py` | 388 | R3: the configuration change and its closure diff |
| `hostd/server.py` | 327 | the socket, authenticated per frame |
| `hostd/audit.py` | 283 | root-owned, append-only, size-rotated record |
| `hostd/executor.py` | 218 | the one place a process is spawned |
| `hostd/protocol.py` | 194 | `PROTOCOL_VERSION`, `Verb`, `Request`, `ResultFrame` |
| `hostd/main.py` | 114 | the console-script entry point |
| `hostd/files.py` | 99 | the filesystem seam |
| `hostd/actions/` | 4 files | the verb set, which IS the risk taxonomy |
| `hostd/__init__.py` | 85 | the public facade - 39 names, already explicit |
| `hostd/README.md` | - | moves with the package |

Tests that move to `packages/hostd/tests/`:

| File | Note |
|---|---|
| `tests/test_hostd_audit.py` | pure hostd |
| `tests/test_host_actions.py` | pure hostd ENGINE, but imports `scufris.host.run` (open q. 1) |
| `tests/test_nixos_activation.py` | pure hostd R3, same import |

Tests that STAY at the root because they exercise the app over the helper:
`test_host_action_api.py`, `test_host_action_decisions.py`, `test_domain_routers.py`,
`test_host_mcp_server.py`, `test_telegram_approvals.py`, `test_host_digest.py`,
`test_nixos_config_change.py`, `tests/conftest.py`.

Edited:

| File | Why |
|---|---|
| root `pyproject.toml` | drop the `scufris-hostd` script, add the exact pin and the workspace source |
| `packages/hostd/pyproject.toml` | new; declares `pydantic` and (per open q. 1) `scufris-host` |
| `flake.nix` | a second `mkApplication` for the helper - see open question 2 |
| `nix/scufris-hostd.nix` | `package` default re-pointed |
| `scufris/hostclient.py` | `from .hostd.X` -> `from scufris_hostd import X` |
| `scufris/host_actions.py`, `host_approvals.py`, `checks.py`, `hostconfig/changes.py`, `hostconfig/service.py` | same re-point, 1-2 lines each |
| 5 examples, 8 root tests | same re-point |
| `scufris/README.md`, `AGENTS.md` | the module map moves |
| `scripts/check_file_size.py` | `COVERED_ROOTS` must reach `packages/` (746's open q. 6) |

## Data and interfaces

The package's public API is unchanged - `scufris/hostd/__init__.py` already
exports exactly 39 names and every app-side importer already goes through either
it or a named submodule. Nothing is added; the module path changes:

```python
# before                                    # after
from scufris.hostd import HostdEngine       from scufris_hostd import HostdEngine
from scufris.hostd.protocol import Verb     from scufris_hostd.protocol import Verb
```

The new test, which is what replaces the same-wheel guarantee:

```python
def test_hostd_and_app_report_the_same_protocol_version() -> None: ...
```

There is a problem with the obvious implementation. `PROTOCOL_VERSION` appears
in ZERO files outside `scufris/hostd/` today - the app imports the protocol
MODELS, never the number, and `hostclient` performs no version handshake. So the
app has no "reported protocol version" to compare against. What actually needs
guarding is DEPLOYMENT drift: the app's venv and the running root unit resolving
different builds. Two honest shapes, planner picks:

```python
# (a) metadata: the pin is exact, so one venv cannot hold two versions
importlib.metadata.version("scufris") == importlib.metadata.version("scufris-hostd")
# plus a parse of the root pyproject asserting the specifier is `==`, not `>=`.

# (b) handshake: hostclient calls `hello` on connect and refuses a mismatch.
```

(a) is a move-compatible test. (b) is a BEHAVIOR CHANGE and the task forbids one
here. Recommendation: (a), and file (b) as its own task if the deployment split
is ever real.

## Sketches

Illustrative only.

```diff
# pyproject.toml (root)
 [project.scripts]
 scufris = "scufris.__main__:main"
-scufris-hostd = "scufris.hostd.main:main"
 dependencies = [
+    # EXACT, not a range: the two halves speak one protocol version and the
+    # same-wheel guarantee that used to enforce that is gone.
+    "scufris-hostd==0.1.0",
```

```diff
# flake.nix
+        hostdApp = mkApplication {
+          venv = runtimeVenv;
+          package = pythonSet.scufris-hostd;
+        };
         packages = {
           scufris = ...;
+          scufris-hostd = hostdApp;
```

```diff
# nix/scufris-hostd.nix
-      default = self.packages.${pkgs.system}.scufris;
-      defaultText = "scufris.packages.\${system}.scufris";
+      default = self.packages.${pkgs.system}.scufris-hostd;
+      defaultText = "scufris.packages.\${system}.scufris-hostd";
```

## Shape

Today, and the reason the task cannot run first:

```
  scufris/hostd/  ---- imports ---->  scufris/host/  (run, models, storage, units)
       engine.py       Runner, run_command, CommandResult, Outcome
       preview.py      Availability, Runner, nix_cli, list_generations, unit_status
       nixos.py        Availability, Outcome, Runner, nix_cli, Generation
       executor.py     CommandResult, Outcome
       actions/*.py    Runner, nix_cli, Generation, list_generations
       main.py    ----> scufris/logsetup.py  (configure_logging)
```

After, with 748 done first:

```
  core  <-  host  <-  hostd  <-  hostctl  <-  scufris
                        ^
                        |  a unix socket, the one real process boundary
                     root unit
```

Deployment after the split:

```
  packages.scufris        -> bin/scufris        (venv holds BOTH distributions)
  packages.scufris-hostd  -> bin/scufris-hostd  (same venv, different mkApplication)
  nix/scufris-hostd.nix   -> ExecStart = ${scufris-hostd}/bin/scufris-hostd
```

## Consequences and open questions

Cost: one distribution, one lock entry, an exact pin that must be bumped in
lockstep at every release, and a second `mkApplication` in `flake.nix`. Bought:
the only externally observable boundary in the tree becomes a real one, and the
carve path is proven on code nobody is changing.

Forecloses: shipping the helper and the app as one artifact. The exact pin keeps
them from drifting inside one venv; nothing stops an operator from installing two
versions on the same machine, and nothing did before either.

**Open questions for the planner.** 1 and 2 are blocking.

1. **`hostd` is not import-clean, so it cannot move first or alone.** Six
   modules import `scufris.host.run` / `.models` / `.storage` / `.units`
   (`engine.py:33`, `preview.py:26-29`, `nixos.py:41-43`, `executor.py:25`,
   `actions/validate.py:18`, `actions/plans.py:12-13`), and `main.py:17` imports
   `scufris.logsetup`. The epic's dependency table has no `hostd -> host` edge
   and its sequencing puts `hostd` at p104, ahead of `host` at p103. Three ways out:
   - **Reorder: run 20260803-214748 (`host`) before this task**, and amend the
     epic graph to `core <- host <- hostd <- hostctl`. Still acyclic, still
     honest about privilege - `host.run` is read-only command plumbing that the
     root helper legitimately reuses. **Recommended**, and it is a one-line edit
     to the epic plus a priority swap.
   - Copy the ~200 lines of `host/run.py` into `hostd`. Rejected: duplicating
     `CommandResult`/`Outcome` puts two definitions on the wire boundary the
     rest of this task exists to protect.
   - Hoist `host/run.py` into `core`. Rejected: it shells out to `nix` and
     `systemctl`; putting it in `core` is exactly the junk-drawer decay
     `test_core_is_domain_free` is meant to catch.

   `logsetup.py` (87 lines, no scufris imports, imported by 9 root modules) is a
   separate small decision: move it to `core` in 20260803-214746, or let
   `hostd/main.py` configure its own logging. Recommend core.

2. **`packages.scufris` stops carrying `bin/scufris-hostd`.**
   `mkApplication` (flake.nix:117) builds its output from the STRUCTURE of
   `package`, symlinking only paths that exist in `pythonSet.scufris`'s own
   derivation - verified in pyproject-nix's `mk-application.py`. Move the console
   script to another distribution and `${pkgs.scufris}/bin/scufris-hostd`
   disappears, which is exactly what `nix/scufris-hostd.nix:59` defaults to and
   what its `ExecStart` execs. The NixOS module breaks at BUILD time, not at
   runtime, so `nix build .#checks...scufris-hostd-vm-test` is the proof - it is
   already in the Definition of Done. Fix: export a second `mkApplication` and
   re-point the module default.

3. **`test_hostd_and_app_report_the_same_protocol_version` has no app-side
   subject.** See Data and interfaces. Recommendation (a).

4. **Two hostd test modules import `scufris.host.run`.**
   `test_host_actions.py:23` and `test_nixos_activation.py:38` use
   `FakeRunner`/`CommandResult`/`ok_result` as doubles. Under recommendation 1
   this is fine (hostd depends on host anyway); under any other resolution those
   two files cannot move.

5. **`hostclient.py` imports three hostd SUBMODULES** (`.hostd.actions`,
   `.hostd.audit`, `.hostd.protocol`). The epic's rule only forbids a sibling's
   `models` / `repo`, so this is legal - but `scufris_hostd/__init__.py` already
   re-exports every name involved. Re-pointing through the facade costs nothing
   and makes the boundary greppable. Recommend the facade.

6. **`scufris/hostd/actions/models.py` exists.** Under the epic rule no sibling
   may import it. Nothing outside `hostd` does today - keep it that way, and let
   `test_no_package_imports_a_sibling_private_module` (746) cover it. This is
   the first task where that test stops being vacuous, which is 746's open
   question 5.

7. **Version bumping.** An exact pin means the release procedure must bump two
   `pyproject.toml` versions together. `scripts/check-release-ready.sh` and
   `docs/RELEASING.md` do not know about a second distribution yet.
