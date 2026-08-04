"""A reviewed commit becomes the running system, reversibly - the client's half.

Two layers, and neither needs the app:

- the REPOSITORY, against a real temporary git repo, to prove the flow cannot
  write to it and that a ref or a repository outside the allowed one is refused;
- the STORE, against a file-backed database, for the bound the app cannot reach
  through HTTP without a hundred builds.

The APP's half - the build pipeline driven over HTTP against a real hostd
socket - is `tests/test_nixos_config_change.py`. The helper's plan, preview,
rollback and apply are `packages/hostd/tests/test_nixos_activation.py`.

Nothing here imports `scufris`. That is the property this file exists to hold
as much as any assertion in it: the configuration change flow is the client's,
and the client is a distribution that does not know the app.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from scufris_core import Base, Database, open_database
from scufris_hostctl import (
    ChangeState,
    ConfigChange,
    ConfigChangeRefused,
    ConfigChangeStore,
    Resolved,
    build_argv,
    flake_url,
    render_change,
    resolve,
    toplevel_from,
)

# A store path standing in for what a build produced. A shape, not a contract.
BUILT = "/nix/store/c0z2q4wl5m7dnpx9rsv0abcdfghijklm-nixos-system-nixos-26.11"

#: The tables `scufris_hostctl.models` declares, by name.
OWNED_TABLES = ("host_action", "config_change")


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    """A file-backed state database holding this package's tables.

    Owned here rather than borrowed: `packages/*/tests` does not see the root's
    `tests/conftest.py`, and a package test that needed the app's fixtures would
    be an app test in the wrong directory. A local `conftest.py` is not the way
    to own it either - pytest imports every `conftest.py` under the same
    top-level `conftest` name, so one here shadows the root's for the whole run.

    The tables are created straight from the shared metadata rather than by
    running Alembic. The migration environment ships with the root distribution,
    and a package suite that ran it would depend on the app to test a store
    bound. `tests/test_db_migrations.py::test_every_package_model_is_registered`
    is what proves the two agree.

    File-backed rather than `:memory:`, matching the root suite: a store test
    that reopens the file is a proof `:memory:` cannot carry.
    """
    db = open_database(tmp_path)
    try:
        Base.metadata.create_all(
            db.engine,
            tables=[Base.metadata.tables[name] for name in OWNED_TABLES],
        )
        yield db
    finally:
        db.close()


@pytest.fixture
def config_repo(tmp_path: Path) -> Path:
    """A real git repository with a flake, committed on a branch."""
    if shutil.which("git") is None:  # pragma: no cover - git is in the dev shell
        pytest.skip("git is not on PATH")
    repo = tmp_path / "nix.dotfiles"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            env={
                # PATH from the environment, not a guess: `nix flake check` runs
                # pytest in a sandbox where /usr/bin does not exist and git comes
                # from the check's own nativeBuildInputs. HOME is still
                # overridden, which is what keeps the repo hermetic.
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(tmp_path),
                "GIT_AUTHOR_NAME": "t",
                "GIT_AUTHOR_EMAIL": "t@e",
                "GIT_COMMITTER_NAME": "t",
                "GIT_COMMITTER_EMAIL": "t@e",
            },
        ).stdout.strip()

    git("init", "-q", "-b", "master")
    (repo / "flake.nix").write_text("{ outputs = _: {}; }\n")
    git("add", "flake.nix")
    git("commit", "-qm", "initial")
    git("checkout", "-qb", "config/add-ripgrep")
    (repo / "packages.nix").write_text("[ ripgrep ]\n")
    git("add", "packages.nix")
    git("commit", "-qm", "feat: add ripgrep")
    return repo


def _repo_state(repo: Path) -> tuple[str, str, str]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True
        ).stdout

    return (
        git("status", "--porcelain"),
        git("log", "--all", "--format=%H"),
        git("branch", "--format=%(refname)"),
    )


def test_nixos_change_never_writes_to_the_config_repo(config_repo: Path) -> None:
    """Cleanliness is structural here, not something a teardown achieves.

    The build addresses the repository as `git+file://...?rev=`, so nix reads the
    tree from the commit: there is no worktree to leave behind, no `result`
    symlink, no lock-file write and no commit. This replaces the planned
    `test_rejected_nixos_proposal_leaves_repo_clean` - with the edit owned by the
    project flow, there is nothing for a rejected proposal to clean up.
    """
    before = _repo_state(config_repo)

    main, resolved = resolve(config_repo, "config/add-ripgrep")
    url = flake_url(main, resolved)
    argv = build_argv(url, "nixos")

    assert main == config_repo
    assert resolved.rev and resolved.subject == "feat: add ripgrep"
    # The revision is pinned INTO the flake reference, so the working tree is not
    # what gets built.
    assert f"rev={resolved.rev}" in url
    assert "ref=config/add-ripgrep" in url
    assert argv[0] == "nix" and "build" in argv
    for flag in ("--no-link", "--no-update-lock-file", "--no-write-lock-file"):
        assert flag in argv
    # Nothing in the argv names the working tree as a source.
    assert not [part for part in argv if part == str(config_repo)]

    assert _repo_state(config_repo) == before
    assert not (config_repo / "result").exists()
    # And no worktree was created anywhere for this.
    listed = subprocess.run(
        ["git", "-C", str(config_repo), "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
    ).stdout
    assert listed.count("worktree ") == 1


def test_an_uncommitted_edit_is_reported_as_not_in_the_build(
    config_repo: Path,
) -> None:
    """An agent that edited but did not commit must be told, not left guessing."""
    (config_repo / "packages.nix").write_text("[ ripgrep fd ]\n")

    _main, resolved = resolve(config_repo, "HEAD")

    assert resolved.uncommitted == ["packages.nix"]

    text = render_change(ConfigChange(id="x", resolved=resolved, attr="nixos"))
    assert "are NOT in this build" in text
    assert "packages.nix" in text


def test_a_ref_that_does_not_exist_is_refused_by_name(config_repo: Path) -> None:
    with pytest.raises(ConfigChangeRefused) as refused:
        resolve(config_repo, "config/typo")
    assert "does not name a commit" in str(refused.value)


@pytest.mark.parametrize("hostile", ["-c", "--upload-pack=x", "a..b", "master;rm"])
def test_a_ref_outside_the_charset_is_refused(config_repo: Path, hostile: str) -> None:
    """A ref reaches a git argv, so it is charset-validated like a unit name."""
    with pytest.raises(ConfigChangeRefused):
        resolve(config_repo, hostile)


def test_a_repository_other_than_this_host_s_configuration_is_refused(
    config_repo: Path, tmp_path: Path
) -> None:
    """Which revision to build is a caller's choice; which repository is not.

    Without this, an agent could commit its own flake anywhere it can write and
    have the server build and propose THAT as the system - the same shape as
    handing over a store path, one step removed.
    """
    other = tmp_path / "mine"
    other.mkdir()
    subprocess.run(["git", "-C", str(other), "init", "-q"], check=True)
    (other / "flake.nix").write_text("{ outputs = _: {}; }\n")

    with pytest.raises(ConfigChangeRefused) as refused:
        resolve(other, "HEAD", allowed=config_repo)
    assert "is not it" in str(refused.value)

    # A WORKTREE of the allowed repository is fine - that is where an agent
    # works - because the check is on the main repository the commits live in.
    worktree = tmp_path / "wt"
    subprocess.run(
        ["git", "-C", str(config_repo), "worktree", "add", "-q", str(worktree)],
        check=True,
        capture_output=True,
    )
    main, resolved = resolve(worktree, "HEAD", allowed=config_repo)
    assert main == config_repo
    assert resolved.rev


def test_a_ref_of_head_is_recorded_as_the_branch_it_is(config_repo: Path) -> None:
    """ "ref: HEAD @ 3af39d5" in an approval prompt tells the operator nothing."""
    _main, resolved = resolve(config_repo, "HEAD")

    assert resolved.ref == "config/add-ripgrep"
    assert "ref=config/add-ripgrep" in flake_url(config_repo, resolved)


def test_a_directory_that_is_not_a_flake_is_refused(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ConfigChangeRefused):
        resolve(plain, "HEAD")


def test_only_a_store_path_is_taken_from_a_build() -> None:
    """The out path is read from stdout, and only if it IS a store path."""
    assert toplevel_from(f"warning: dirty tree\n{BUILT}\n") == BUILT
    assert toplevel_from("built nothing\n") == ""
    assert toplevel_from("/etc/passwd\n") == ""


def _stored(change_id: str, state: ChangeState) -> ConfigChange:
    return ConfigChange(
        id=change_id,
        resolved=Resolved(repo="/srv/config", ref="master", rev="0" * 40),
        attr="nixos",
        state=state,
    )


def test_the_change_registry_stays_bounded(database: Database) -> None:
    """The bound drops settled changes first, and the oldest when none settled.

    A building change has a live run behind it, so it is never dropped ahead of
    one that has finished. When everything is building the table must still stop
    growing, so the oldest goes anyway.
    """
    store = ConfigChangeStore(database, max_changes=3)

    # The settled change is NOT the oldest, so dropping it is a choice about
    # state rather than about age: a bound that only looked at `seq` would take
    # `building-1` here instead.
    store.put(_stored("building-1", ChangeState.BUILDING))
    store.put(_stored("settled", ChangeState.PROPOSED))
    store.put(_stored("building-2", ChangeState.BUILDING))
    store.put(_stored("building-3", ChangeState.BUILDING))
    assert [c.id for c in store.list()] == ["building-3", "building-2", "building-1"]

    # With nothing settled left, the bound falls on the oldest `seq` anyway
    # rather than letting the table grow.
    store.put(_stored("building-4", ChangeState.BUILDING))
    assert [c.id for c in store.list()] == ["building-4", "building-3", "building-2"]
