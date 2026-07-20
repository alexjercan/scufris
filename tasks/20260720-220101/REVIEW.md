# Review

## Round 1

- VERDICT: APPROVE
- REVIEWER: out-of-context

What I tried to break: I ran the full suite from the worktree root under the nix
devShell (`python -m pytest -q`) and it passed with no false-fire from the guard.
I reproduced the intended failure by running pytest from `/tmp` against the
worktree's tests, and the guard fired at conftest import with a clean
`RuntimeError` naming both the resolved package root and the cwd and pointing at
`python -m pytest` - exactly the fail-fast collection error we want, and the
correct diagnosis of the import-shadowing trap. ruff and mypy are clean on the
changed conftest. The one thing I actively broke: running `python -m pytest`
from a subdirectory of the repo (`cd tests && python -m pytest .`) also trips the
guard, because the condition only treats cwd as valid when it equals pkg_root or
is an ancestor of it, whereas a subdirectory is a descendant. So the guard is
"run from the repo root only", not "run from anywhere inside the repo". I judge
this acceptable: AGENTS.md documents the workflow as running the QA commands from
the repo root, `python -m pytest` puts cwd first on sys.path so from the root the
import always resolves to the worktree, and a false-fire is a loud, self-
explaining error (not a silent wrong-tree run), so the failure mode is safe. The
guard does not interfere with the main-checkout workflow: from the main checkout
root, pkg_root == cwd and it stays quiet.

The guard logic is sound for its stated contract. `Path(scufris.__file__).resolve().parent.parent`
is the correct package root: `scufris/__init__.py` -> `.parent` is `scufris/` ->
`.parent.parent` is the repo root, which is what `cwd` is compared against. The
`.resolve()` on both sides normalizes symlinks so a sprout worktree and its
source compare correctly. The AGENTS.md documentation is accurate: it correctly
states that the console-script `pytest` does not put cwd first on sys.path, that
this can import scufris from the main checkout, and that conftest fails fast with
a pointer to `python -m pytest`.

- [x] R1.1 (NIT) tests/conftest.py:29 - The guard rejects running from a repo
  subdirectory (e.g. `cd tests && python -m pytest .`), since the condition
  admits cwd only when it equals or is an ancestor of the package root, not a
  descendant. Given tests are run from the repo root here and a false-fire is a
  loud, safe error, this is a minor ergonomic limitation, not a correctness bug.
  If broadening is desired, also accept `_pkg_root in _cwd.parents`.
  - Response: fixed. The condition is now
    `_pkg_root != _cwd and _pkg_root not in _cwd.parents`, so it stays quiet when
    the package root is the cwd OR an ancestor of it (repo root or any
    subdirectory), and fires only when scufris resolves from an unrelated tree.
    Verified: root run passes, subdirectory run (`cd tests && python -m pytest .`)
    now passes with no false-fire, and the `/tmp` invariant-violation still fires.

## Round 2 (implementer note)

Addressed the R1.1 NIT (already APPROVEd in round 1) with the reversed condition
above. Re-verified root/subdirectory/foreign-cwd behavior and ruff+mypy clean.
Verdict stands: APPROVE.
