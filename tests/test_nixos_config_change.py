"""A reviewed commit becomes the running system, reversibly - the app's half.

What is left here is the APP's build pipeline: resolve a ref, build a commit,
propose the result, survive a restart - every one of them driven over HTTP
against a real hostd socket, so every one of them needs `create_app`.

The two layers that do NOT need the app moved out with the code they cover:
the repository proofs and the store bound are
``packages/hostctl/tests/test_config_change_service.py``. The helper's plan,
preview, rollback and apply are
``packages/hostd/tests/test_nixos_activation.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import ORIGIN, _Helper, _login, _settings
from fastapi.testclient import TestClient

from scufris.app import create_app
from scufris.auth import CSRF_HEADER
from scufris.db import open_database
from scufris_host import (
    NIX_FEATURES,
    Collector,
    CommandResult,
    FakeRunner,
    Outcome,
    ok_result,
)
from scufris_hostctl import (
    ChangeState,
    ConfigChangeStore,
)
from scufris_hostd import FakeExecutor

# These three used to be imported from `test_host_actions`, which now lives in
# `packages/hostd/tests/`. They are FakeRunner keys and fixture store paths -
# shapes, not a contract - so this suite owns its own rather than reaching
# across a distribution boundary into another package's tests.
NIX = " ".join(["nix", *NIX_FEATURES])
RUNNING = "/nix/store/bnfi69bsjhaj4jgp42jk9ys6y80pb9qh-nixos-system-nixos-26.11"
BUILT = "/nix/store/c0z2q4wl5m7dnpx9rsv0abcdfghijklm-nixos-system-nixos-26.11"

# What the host looks like AFTER an activation: a new generation, and
# /run/current-system pointing at what was built.
GENERATIONS_AFTER_SWITCH = ok_result(
    json.dumps(
        [
            {
                "generation": 192,
                "date": "2026-07-29 12:30:00",
                "nixosVersion": "26.11",
                "current": True,
            },
            {
                "generation": 191,
                "date": "2026-07-29 10:00:00",
                "nixosVersion": "26.11",
                "current": False,
            },
            {
                "generation": 190,
                "date": "2026-07-28 10:00:00",
                "nixosVersion": "26.11",
                "current": False,
            },
        ]
    )
)


def _switched(host: _Helper) -> None:
    """Move the faked host to where a successful activation would have left it."""
    host.runner.results["nixos-rebuild list-generations"] = GENERATIONS_AFTER_SWITCH
    host.files.links["/run/current-system"] = BUILT
    host.files.links["/nix/var/nix/profiles/system-192-link"] = BUILT


# --- the app: build a commit, then propose it ----------------------------


class _BuildExecutor(FakeExecutor):
    """A build that prints a store path, and then whatever the apply needs."""

    def __init__(self, *, out: str = BUILT, fail: bool = False, hang: bool = False):
        super().__init__()
        self.out = out
        self.fail = fail
        self.hang_build = hang

    async def __call__(  # type: ignore[override]
        self, argv: list[str], *, timeout: float, on_output: Any
    ) -> CommandResult:
        self.calls.append(list(argv))
        if argv[0] == "nix" and "build" in argv:
            on_output("stderr", "building nixos-system...\n")
            self.started.set()
            if self.hang_build:
                await asyncio.sleep(3600)
            if self.fail:
                # run_action streams stderr through on_output as it arrives, so a
                # fake that only returned it would exercise a path the real
                # executor does not have.
                on_output("stderr", "error: attribute 'ripgrp' missing\n")
                return CommandResult(
                    argv=argv,
                    outcome=Outcome.FAILED,
                    stdout="",
                    stderr="error: attribute 'ripgrp' missing\n",
                    returncode=1,
                )
            return CommandResult(
                argv=argv, outcome=Outcome.OK, stdout=f"{self.out}\n", returncode=0
            )
        return CommandResult(argv=argv, outcome=Outcome.OK, returncode=0)


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


def test_the_attribute_probe_does_not_delay_the_request(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,  # noqa: F811
    make_client: Callable[[Any], TestClient],  # noqa: F811
    config_repo: Path,
) -> None:
    """An unknown host attribute fails the CHANGE, not the request that started it.

    The probe is a full flake evaluation (measured 6.4s warm, slower cold) and
    the MCP tool's API timeout is 15s, so probing before answering made the one
    call an agent always makes report a timeout for a build that was running.
    """
    app = _app(tmp_path, fake_collector, helper, config_repo)
    client = make_client(app)
    csrf = _login(client)

    resp = _post(
        client,
        csrf,
        "/api/host/config/changes",
        ref="config/add-ripgrep",
        attr="not-this-host",
    )

    assert resp.status_code == 201, resp.text
    change = _settle(client, csrf, resp.json()["id"], want="failed")
    assert "no nixosConfigurations.not-this-host" in change["error"]
    # It says which ones exist, and it never reached a build.
    assert "nixos" in change["error"]
    assert change["action_id"] == ""


# --- the app over HTTP ----------------------------------------------------


def _app(
    tmp_path: Path,
    fake_collector: Collector,
    host: _Helper,
    config_repo: Path,
    *,
    build: _BuildExecutor | None = None,
) -> Any:
    from scufris_hostctl import ConfigChangeBuilder

    return create_app(
        collector=fake_collector,
        settings=_settings(
            tmp_path,
            host,
            host_config_repo=config_repo,
            host_config_attr="nixos",
        ),
        # The attribute probe and the git reads go through the runner; the BUILD
        # goes through the executor, which is the only thing being faked.
        config_builder=ConfigChangeBuilder(
            runner=_config_runner(),
            executor=build or _BuildExecutor(),
        ),
    )


def _config_runner(**overrides: CommandResult) -> FakeRunner:
    """git reads run for real; only the flake evaluation is canned."""
    results: dict[str, CommandResult] = {
        f"{NIX} eval": ok_result(json.dumps(["nixos"])),
    }
    results.update(overrides)
    real = FakeRunner(results=results)

    def run(argv: list[str], *, timeout: float = 30.0) -> CommandResult:
        if argv and argv[0] == "git":
            from scufris_host import run_command

            return run_command(argv, timeout=timeout)
        return real(argv, timeout=timeout)

    return run  # type: ignore[return-value]


def _post(client: TestClient, csrf: str, path: str, **body: Any) -> Any:
    return client.post(path, json=body, headers={"Origin": ORIGIN, CSRF_HEADER: csrf})


def _settle(client: TestClient, csrf: str, change_id: str, *, want: str) -> Any:
    """Poll a change until it leaves `building`."""
    for _ in range(200):
        change = client.get(f"/api/host/config/changes/{change_id}").json()
        if change["state"] != "building":
            assert change["state"] == want, change
            return change
        import time as _time

        _time.sleep(0.05)
    raise AssertionError("the change never settled")


def test_propose_refuses_a_caller_supplied_toplevel(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,  # noqa: F811
    make_client: Callable[[Any], TestClient],  # noqa: F811
    config_repo: Path,
) -> None:
    """If the caller picks the store path, the closure diff describes its choice.

    So the generic propose surface has no `activate` at all, and the refusal
    names the endpoint that does build one.
    """
    client = make_client(_app(tmp_path, fake_collector, helper, config_repo))
    csrf = _login(client)

    resp = _post(
        client,
        csrf,
        "/api/host/actions",
        kind="activate",
        args={"toplevel": BUILT},
    )

    assert resp.status_code == 422, resp.text
    assert "not proposed directly" in resp.text
    assert "/api/host/config/changes" in resp.text
    # Nothing reached the helper, so there is no proposal to approve.
    assert client.get("/api/host/actions").json() == []
    assert helper.executor.calls == []


def test_nixos_build_failure_blocks_activation(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,  # noqa: F811
    make_client: Callable[[Any], TestClient],  # noqa: F811
    config_repo: Path,
) -> None:
    """A configuration that does not build has no route to activation.

    Not because a check refuses it: because the thing an approval would act on
    does not exist. The log comes back so the failure is diagnosable.
    """
    app = _app(
        tmp_path, fake_collector, helper, config_repo, build=_BuildExecutor(fail=True)
    )
    client = make_client(app)
    csrf = _login(client)

    resp = _post(client, csrf, "/api/host/config/changes", ref="config/add-ripgrep")
    assert resp.status_code == 201, resp.text
    change = _settle(client, csrf, resp.json()["id"], want="failed")

    assert change["action_id"] == ""
    assert "did not build" in change["error"]
    assert "ripgrp" in change["log_tail"]
    # No proposal exists, so there is nothing an approval could act on - and the
    # helper never ran anything.
    assert client.get("/api/host/actions").json() == []
    assert helper.executor.calls == []


def test_nixos_change_builds_diffs_switches_and_rolls_back(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,  # noqa: F811
    make_client: Callable[[Any], TestClient],  # noqa: F811
    config_repo: Path,
) -> None:
    """The whole contract, end to end, over HTTP against the real helper.

    resolve a ref -> build the commit -> a closure-diff preview -> the operator
    approves -> the profile moves and the system switches -> the generation it
    replaced is recorded as the way back -> roll back to it.
    """
    app = _app(tmp_path, fake_collector, helper, config_repo)
    client = make_client(app)
    csrf = _login(client)

    resp = _post(client, csrf, "/api/host/config/changes", ref="config/add-ripgrep")
    assert resp.status_code == 201, resp.text
    change = _settle(client, csrf, resp.json()["id"], want="proposed")

    # The build was addressed at the commit, and it produced a real store path.
    assert change["resolved"]["rev"]
    assert change["toplevel"] == BUILT
    assert change["action_id"]

    action = client.get(f"/api/host/actions/{change['action_id']}").json()
    assert action["proposal"]["risk"] == "r3"
    assert action["proposal"]["kind"] == "activate"
    body = "\n".join(action["proposal"]["preview"]["lines"])
    assert "closure diff" in body
    assert "ripgrep: (none) -> 14.1.1" in body
    assert change["resolved"]["rev"] in body
    # Approving is what runs it. Until then, nothing has.
    assert helper.executor.calls == []

    approved = _post(client, csrf, f"/api/host/actions/{change['action_id']}/approve")
    assert approved.status_code == 200, approved.text

    for _ in range(200):
        action = client.get(f"/api/host/actions/{change['action_id']}").json()
        if action["result"] is not None:
            break
        import time as _time

        _time.sleep(0.05)
    assert action["result"] is not None and action["result"]["ok"], action
    assert action["result"]["steps_completed"] == 2

    # The profile moved first, and then THAT path was switched to.
    assert helper.executor.calls[0] == [
        "nix-env",
        "--profile",
        "/nix/var/nix/profiles/system",
        "--set",
        BUILT,
    ]
    assert helper.executor.calls[1][-2:] == [
        f"{BUILT}/bin/switch-to-configuration",
        "switch",
    ]

    # The way back was recorded at approval time, as a generation number.
    assert action["proposal"]["reversal"]["possible"]
    assert action["proposal"]["reversal"]["args"] == {"generation": 191}

    # The machine has now switched, so the faked host moves with it: generation
    # 192 is current and /run/current-system points at what was built. A rollback
    # proposed against a host still claiming to run 191 would be testing a
    # machine that cannot exist.
    _switched(helper)

    reverted = _post(client, csrf, f"/api/host/actions/{change['action_id']}/revert")
    assert reverted.status_code == 201, reverted.text
    rollback = reverted.json()
    assert rollback["proposal"]["kind"] == "rollback"
    assert rollback["proposal"]["args"]["generation"] == 191
    # The rollback resolved its own store path from the profile.
    assert rollback["proposal"]["args"]["toplevel"] == RUNNING

    # `id` is a property on the record, so the wire shape carries it on the
    # proposal.
    rollback_id = rollback["proposal"]["id"]
    approved = _post(client, csrf, f"/api/host/actions/{rollback_id}/approve")
    assert approved.status_code == 200, approved.text
    for _ in range(200):
        rollback = client.get(f"/api/host/actions/{rollback_id}").json()
        if rollback["result"] is not None:
            break
        import time as _time

        _time.sleep(0.05)
    assert rollback["result"] is not None and rollback["result"]["ok"], rollback
    assert helper.executor.calls[2] == [
        "nix-env",
        "--profile",
        "/nix/var/nix/profiles/system",
        "--switch-generation",
        "191",
    ]

    # And the helper's own record carries the whole story, including the
    # revision that produced what is now running.
    audit = client.get("/api/host/audit?limit=100").json()
    applied = [r for r in audit if r["event"] == "applied"]
    assert [r["kind"] for r in applied] == ["activate", "rollback"]
    assert applied[0]["args"]["rev"] == change["resolved"]["rev"]
    assert applied[0]["args"]["toplevel"] == BUILT
    assert applied[0]["restore_point"]
    assert "191" in applied[0]["reversal"]


def test_a_configuration_change_survives_a_restart(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,  # noqa: F811
    make_client: Callable[[Any], TestClient],  # noqa: F811
    config_repo: Path,
) -> None:
    """A built change and the proposal it produced outlive the process.

    The build is the expensive half - minutes to hours - and the proposal it
    produced is what the operator is being asked about. A restart that answers
    "there was never any such change" makes the operator start it again.
    """
    # `create_app` memoizes its handle process-wide, so the first client must
    # exit - and hence cannot be a `make_client` one, whose ExitStack unwinds at
    # teardown - before the restarted app is built.
    first = _app(tmp_path, fake_collector, helper, config_repo)
    with TestClient(first) as client:
        csrf = _login(client)

        resp = _post(client, csrf, "/api/host/config/changes", ref="config/add-ripgrep")
        assert resp.status_code == 201, resp.text
        before = _settle(client, csrf, resp.json()["id"], want="proposed")
        assert before["toplevel"] == BUILT and before["action_id"]

    second = _app(tmp_path, fake_collector, helper, config_repo)
    assert second.state.db is not first.state.db
    restarted = make_client(second)
    _login(restarted)

    after = restarted.get(f"/api/host/config/changes/{before['id']}")
    assert after.status_code == 200, after.text
    assert after.json()["state"] == "proposed"
    assert after.json()["toplevel"] == BUILT
    assert after.json()["action_id"] == before["action_id"]
    assert [c["id"] for c in restarted.get("/api/host/config/changes").json()] == [
        before["id"]
    ]


def test_a_build_interrupted_by_a_restart_does_not_block_the_repo(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,  # noqa: F811
    make_client: Callable[[Any], TestClient],  # noqa: F811
    config_repo: Path,
) -> None:
    """Durability must not turn a crash into a permanent 409.

    In memory, a build killed by a restart simply vanished with the process. On
    a row it would stay `building` forever, `building_for` would keep answering
    it, and the documented escape - cancel - cannot clear it, because cancelling
    needs a live supervisor run and the supervisor was restarted too. So the
    startup sweep fails it, with a reason, and the repository is buildable
    again. DECISION.md 2.
    """
    # The first process is deliberately left open: a graceful shutdown writes
    # `cancelled`, not `building`, so the row this sweep exists for cannot be
    # produced that way - see 20260803-014401 DECISION.md 1 and task
    # 20260803-113000. What is proven here is the live-process case.
    client = make_client(
        _app(
            tmp_path,
            fake_collector,
            helper,
            config_repo,
            build=_BuildExecutor(hang=True),
        )
    )
    csrf = _login(client)

    resp = _post(client, csrf, "/api/host/config/changes", ref="config/add-ripgrep")
    assert resp.status_code == 201, resp.text
    interrupted = resp.json()["id"]
    assert client.get(f"/api/host/config/changes/{interrupted}").json()["state"] == (
        "building"
    )

    # The process is replaced while that build is still running.
    restarted = make_client(
        _app(
            tmp_path,
            fake_collector,
            helper,
            config_repo,
            build=_BuildExecutor(hang=True),
        )
    )
    csrf = _login(restarted)

    swept = restarted.get(f"/api/host/config/changes/{interrupted}").json()
    assert swept["state"] == "failed", swept
    assert "restart" in swept["error"]
    assert swept["action_id"] == ""

    # And the repository takes a new build rather than a 409 nothing can clear.
    again = _post(restarted, csrf, "/api/host/config/changes", ref="master")
    assert again.status_code == 201, again.text

    stopped = _post(
        restarted, csrf, f"/api/host/config/changes/{again.json()['id']}/cancel"
    )
    assert stopped.status_code == 200, stopped.text
    _settle(restarted, csrf, again.json()["id"], want="cancelled")


def test_a_building_row_orphaned_by_a_crash_is_swept_at_startup(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,  # noqa: F811
    make_client: Callable[[Any], TestClient],  # noqa: F811
    config_repo: Path,
) -> None:
    """A process killed mid-build leaves a `building` row with nothing behind it.

    SIGKILL runs no handler: the row keeps the state and the empty error the
    build wrote when it started, and the run that would have moved it is gone
    with the process. Nothing in the next process can notice that except the
    startup sweep, so this is the state `abandon_builds` exists for.
    """
    # The row is re-established through the store rather than built over HTTP
    # because no clean shutdown can produce it: the build generator's
    # cancellation handler writes `cancelled` before re-raising, so tearing the
    # first process down leaves `cancelled`, which the sweep neither touches nor
    # needs to. Asserted below rather than assumed. Do not "simplify" this back
    # into a second HTTP build - see task 20260803-113000 and 20260803-014401
    # DECISION.md 1.
    first = _app(
        tmp_path, fake_collector, helper, config_repo, build=_BuildExecutor(hang=True)
    )
    with TestClient(first) as client:
        csrf = _login(client)

        resp = _post(client, csrf, "/api/host/config/changes", ref="config/add-ripgrep")
        assert resp.status_code == 201, resp.text
        orphaned = resp.json()["id"]
        assert client.get(f"/api/host/config/changes/{orphaned}").json()["state"] == (
            "building"
        )

    db = open_database(tmp_path)
    try:
        store = ConfigChangeStore(db)
        settled = store.get(orphaned)
        assert settled.state is ChangeState.CANCELLED, settled

        # Everything else on the row is what the real pipeline wrote; only the
        # state and the error move to where a kill would have left them.
        settled.state = ChangeState.BUILDING
        settled.error = ""
        store.put(settled)
    finally:
        db.close()

    restarted = make_client(_app(tmp_path, fake_collector, helper, config_repo))
    _login(restarted)

    swept = restarted.get(f"/api/host/config/changes/{orphaned}").json()
    assert swept["state"] == "failed", swept
    assert "restart" in swept["error"]
    assert swept["action_id"] == ""


def test_concurrent_nixos_proposals_are_serialized(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,  # noqa: F811
    make_client: Callable[[Any], TestClient],  # noqa: F811
    config_repo: Path,
) -> None:
    """One build per repository, refused rather than quietly queued.

    A queued NixOS build sits for an hour with no visible reason, and two builds
    of one repository contend for the same evaluation and store.
    """
    app = _app(
        tmp_path, fake_collector, helper, config_repo, build=_BuildExecutor(hang=True)
    )
    client = make_client(app)
    csrf = _login(client)

    first = _post(client, csrf, "/api/host/config/changes", ref="config/add-ripgrep")
    assert first.status_code == 201, first.text
    second = _post(client, csrf, "/api/host/config/changes", ref="master")

    assert second.status_code == 409, second.text
    assert first.json()["id"] in second.text
    assert "already running" in second.text
    # The first one is still the only change, and still building.
    listed = client.get("/api/host/config/changes").json()
    assert [c["id"] for c in listed] == [first.json()["id"]]
    assert listed[0]["state"] == "building"

    # Cancelling it is not an operator-only act - a build holds no privilege -
    # and it leaves no proposal behind.
    stopped = _post(
        client, csrf, f"/api/host/config/changes/{first.json()['id']}/cancel"
    )
    assert stopped.status_code == 200, stopped.text
    change = _settle(client, csrf, first.json()["id"], want="cancelled")
    assert change["action_id"] == ""
    assert client.get("/api/host/actions").json() == []
