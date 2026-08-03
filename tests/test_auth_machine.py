"""Machine callers: what the agent's own credential may do, and what it never carries.

The app's MCP tool subprocesses hold a bearer token, so these assert both
halves: the token reaches the API under auth, and nothing that spawns a
subprocess hands a secret to the model. The environment is asserted as it is
actually handed to the process rather than as a declared env dict - inspecting
the dict is vacuous while the secret travels through ``os.environ``.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_auth import _free_port, _settings

from scufris.app import create_app
from scufris.config import SECRET_ENV_VARS, Settings
from scufris_host import Collector

# --- machine callers (DoD) --------------------------------------------------


def test_mcp_tools_reach_the_api_under_auth(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MCP tool servers and the in-process operator console call this app's
    own HTTP API with no cookie. With authentication on they must still get
    through, via the per-process bearer token - and a caller WITHOUT the token
    must not.

    Driven over a REAL uvicorn socket because that is the shape the tool takes
    (a blocking httpx call looping back to this server); ASGITransport cannot
    exercise it. Mirrors test_app::test_tool_console_self_loopback."""
    import httpx
    import uvicorn

    from scufris import mcp_common

    port = _free_port()
    settings = _settings(tmp_path, port=port)
    app = create_app(collector=fake_collector, settings=settings)
    token = app.state.api_token
    assert token, "the app must mint a machine token"

    monkeypatch.setenv("SCUFRIS_API_BASE", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("SCUFRIS_API_TOKEN", token)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started, "uvicorn did not start"

        # The real tool helper, with the token in its environment.
        out = mcp_common._api_call("GET", "/api/projects")  # noqa: SLF001
        assert not out.startswith("error:"), out

        # A state-changing tool call needs no CSRF token: the bearer caller has no
        # ambient cookie to be ridden, and requiring one would break every tool.
        created = mcp_common._api_call(  # noqa: SLF001
            "POST", "/api/projects", body={"path": str(tmp_path)}
        )
        assert not created.startswith("error: 401"), created
        assert not created.startswith("error: 403"), created

        # Without the token the same call is refused - the loopback address alone
        # buys nothing.
        monkeypatch.delenv("SCUFRIS_API_TOKEN")
        denied = mcp_common._api_call("GET", "/api/projects")  # noqa: SLF001
        assert denied.startswith("error: 401"), denied

        # A wrong token is refused too.
        monkeypatch.setenv("SCUFRIS_API_TOKEN", "not-the-token")
        forged = mcp_common._api_call("GET", "/api/projects")  # noqa: SLF001
        assert forged.startswith("error: 401"), forged

        # The tool console runs the tool IN THIS PROCESS, and gets its credential
        # from the ContextVar the endpoint sets - NOT from the environment. Clear
        # the env var entirely so only that path can make this work.
        monkeypatch.delenv("SCUFRIS_API_TOKEN", raising=False)

        # And the tool console (in-process, on the server's own loop) still runs.
        resp = httpx.post(
            f"http://127.0.0.1:{port}/api/agent/tools/pending_agents/run",
            json={"args": {}},
            headers={"Authorization": f"Bearer {token}"},
            timeout=8,
        )
        assert resp.status_code == 200
        assert "no agents are waiting" in resp.json()["text"]
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_agent_env_carries_the_machine_token(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """The token has to actually reach the MCP subprocess environment, or the
    tools authenticate with nothing. Asserts the wiring, not just the mint.

    Both audiences matter: the orchestrator's ``scufris`` server and a sub-agent's
    ``agent`` callback server BOTH call the API (lesson
    tool-reachable-by-two-runners-needs-a-test-per-runner).

    Note there is no environment seeding here: the token travels on the app's own
    Settings, so this is the real path. The companion check that it does NOT
    reach the agent CLI is
    ``test_agent_cli_env_does_not_carry_the_machine_token``."""
    from scufris.agent import scufris_mcp_servers

    settings = _settings(tmp_path, den_path=tmp_path / "den")
    app = create_app(collector=fake_collector, settings=settings)
    token = app.state.api_token

    orchestrator = scufris_mcp_servers(settings, is_orchestrator=True)
    by_name = {server.server_id: server.env for server in orchestrator}
    assert by_name["scufris"].get("SCUFRIS_API_TOKEN") == token

    sub_agent = scufris_mcp_servers(settings, agent_id="a1")
    assert sub_agent[0].env.get("SCUFRIS_API_TOKEN") == token

    # The den server does NOT call the API, so it has no business holding a
    # credential for it.
    assert "SCUFRIS_API_TOKEN" not in by_name["den"]


def test_token_matches_is_total_over_any_string() -> None:
    """The comparison helper must never raise, whatever bytes reach it."""
    from scufris.auth import token_matches

    for presented in ("\xff\xfe", "caf\xe9", "\ud800", "ok"):
        assert token_matches(presented, "expected") is False
    assert token_matches("same", "same") is True
    assert token_matches("\xff", "\xff") is True


def test_agent_cli_env_does_not_carry_the_machine_token(
    fake_collector: Collector, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent CLI's environment must NOT hold the dashboard's API credential.

    Everything the model runs inherits that environment - every shell command,
    every sub-agent, and the den MCP server whatever its permission mode. Asserts
    against the env actually handed to the subprocess, not the declared MCP env
    dict: the earlier version of this check inspected the dict and was therefore
    vacuous while `_codex_env` leaked the token through `os.environ` (review round
    1, finding 2).
    """
    from scufris.agent import _codex_env

    settings = _settings(tmp_path)
    app = create_app(collector=fake_collector, settings=settings)
    token = app.state.api_token

    # The mint must not have gone through the environment at all...
    assert os.environ.get("SCUFRIS_API_TOKEN") != token
    # ...and even if something else set it, the CLI env is stripped.
    monkeypatch.setenv("SCUFRIS_API_TOKEN", token)
    assert "SCUFRIS_API_TOKEN" not in _codex_env(settings)


def test_two_apps_do_not_clobber_each_others_machine_token(
    fake_collector: Collector, tmp_path: Path
) -> None:
    """Each app carries its own token, so creating a second one does not lock the
    first one's tools out (review round 1, finding 3)."""
    first_settings = _settings(tmp_path / "a")
    second_settings = _settings(tmp_path / "b")
    first = create_app(collector=fake_collector, settings=first_settings)
    second = create_app(collector=fake_collector, settings=second_settings)

    assert first.state.api_token != second.state.api_token
    assert first_settings.auth_api_token == first.state.api_token
    assert second_settings.auth_api_token == second.state.api_token

    # Each app still accepts its OWN token and refuses the other's.
    for app, own, other in (
        (first, first.state.api_token, second.state.api_token),
        (second, second.state.api_token, first.state.api_token),
    ):
        client = TestClient(app)
        assert (
            client.get(
                "/api/stats", headers={"Authorization": f"Bearer {own}"}
            ).status_code
            == 200
        )
        assert (
            client.get(
                "/api/stats", headers={"Authorization": f"Bearer {other}"}
            ).status_code
            == 401
        )


def test_agent_cli_env_does_not_carry_the_hostd_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The credential for the ROOT helper's socket must not reach the agent CLI.

    Unlike the machine API token, this secret arrives THROUGH the environment -
    an EnvironmentFile is how a sops secret reaches the unit - so it is present
    by construction and "we never put it in os.environ" is not available as a
    defence. The socket is reachable by anything running as this user, so a
    model holding this value can apply host actions with no operator approval
    at all, which is the one thing the whole framework exists to prevent.
    Review round 1, finding R1.3.
    """
    from scufris.agent import _codex_env

    monkeypatch.setenv("SCUFRIS_HOSTD_SECRET", "the-socket-credential")
    settings = _settings(tmp_path)

    assert settings.hostd_secret == "the-socket-credential"  # the app still has it
    assert "SCUFRIS_HOSTD_SECRET" not in _codex_env(settings)


def test_every_secret_setting_is_stripped_from_the_agent_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A credential added to Settings later is covered by THIS test failing.

    The enumeration is the point: a per-secret `pop` is a thing someone has to
    remember, and the one that was forgotten (the hostd secret) was forgotten
    for exactly that reason.
    """
    from scufris.agent import _codex_env
    from scufris.config import SECRET_ENV_VARS, SECRET_FIELD_PATTERN

    secret_fields = [
        name for name in Settings.model_fields if SECRET_FIELD_PATTERN.search(name)
    ]
    assert secret_fields, "the pattern matched nothing; it has drifted"

    undeclared = [
        name
        for name in secret_fields
        if f"SCUFRIS_{name.upper()}" not in SECRET_ENV_VARS
    ]
    assert not undeclared, (
        f"secret-shaped settings not in SECRET_ENV_VARS: {undeclared}. Either "
        "strip them from the agent environment or rename the field."
    )

    # And the strip actually happens, for every name, against a real environment.
    for name in SECRET_ENV_VARS:
        monkeypatch.setenv(name, "a-credential")
    env = _codex_env(_settings(tmp_path))
    leaked = sorted(name for name in SECRET_ENV_VARS if name in env)
    assert not leaked, f"the agent CLI environment carries: {leaked}"


async def test_the_claude_backend_strips_every_secret_from_its_child_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The strip has to hold at EVERY agent spawn, not just codex's.

    Review round 2, R2.1: `_codex_env` was the only stripper, and the claude
    backend spawned with no `env=` at all - so with `SCUFRIS_AGENT_BACKEND=claude`
    the model's shell held the root helper's socket credential and could apply
    host actions with no operator involved. This drives the REAL spawn against a
    stub binary that dumps its own environment, rather than asserting on the
    kwargs of a fake, because the bug was a missing kwarg.
    """
    from scufris.backends import ClaudeBackend

    dump = tmp_path / "child-env"
    stub = tmp_path / "claude-stub"
    stub.write_text(f'#!/bin/sh\nenv > "{dump}"\n')
    stub.chmod(0o755)

    for name in SECRET_ENV_VARS:
        monkeypatch.setenv(name, f"the-{name.lower()}-value")
    settings = _settings(
        tmp_path, claude_bin=str(stub), claude_home=tmp_path / "claude"
    )

    async for _ in ClaudeBackend().stream(settings, "ping"):
        pass

    child = dict(
        line.split("=", 1)  # type: ignore[misc]
        for line in dump.read_text().splitlines()
        if "=" in line
    )
    leaked = sorted(name for name in SECRET_ENV_VARS if name in child)
    assert not leaked, f"the claude CLI environment carries: {leaked}"


def test_no_agent_subprocess_is_spawned_without_the_stripped_environment() -> None:
    """Every spawn in the package either strips the secrets or is declared not to.

    The strip was applied per call site once and a whole backend was missed
    (review round 2, R2.1), so the guard is structural: this walks the AST of the
    package and fails on any ``create_subprocess_*`` that neither passes
    ``agent_subprocess_env`` (nor ``_codex_env``, which wraps it) nor appears in
    the exemption list below. A backend added later fails HERE rather than
    needing someone to remember the kwarg.
    """
    import ast

    # Spawns that are NOT the model's shell, and so do not inherit the strip.
    # Each is exempt for a stated reason, and a new one has to be added here
    # deliberately - which is the review this test exists to force.
    exempt = {
        # Runs `systemctl`/`uptime` as the app to answer a health probe; the
        # output goes to the dashboard, never to a model.
        ("scufris/health.py", "_run"),
        # The ROOT helper executing an argv IT built after operator approval.
        # It is the other side of the boundary, not something the agent drives.
        ("packages/hostd/src/scufris_hostd/executor.py", "run_action"),
    }
    stripping = {"agent_subprocess_env", "_codex_env"}

    # The sweep follows the code out of the root. Every workspace member's
    # source root is walked too, so carving a spawning module into a package
    # does not quietly retire its guard.
    repo = Path(__file__).resolve().parent.parent
    roots = [repo / "scufris", *sorted(repo.glob("packages/*/src/*"))]
    assert all(root.is_dir() for root in roots), roots
    offenders: list[str] = []
    checked = 0
    for path in sorted(p for root in roots for p in root.rglob("*.py")):
        rel = path.relative_to(repo).as_posix()
        tree = ast.parse(path.read_text())
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # `env=<local>` counts when that local was bound from a stripping
            # call in the same function - the shape `login()` uses to build the
            # environment once and spawn twice.
            stripped_locals = {
                target.id
                for node in ast.walk(func)
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in stripping
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            for node in ast.walk(func):
                if not isinstance(node, ast.Call):
                    continue
                callee = node.func
                if not (
                    isinstance(callee, ast.Attribute)
                    and callee.attr.startswith("create_subprocess")
                ):
                    continue
                checked += 1
                if (rel, func.name) in exempt:
                    continue
                env = next((k for k in node.keywords if k.arg == "env"), None)
                source = None
                if env is not None:
                    if isinstance(env.value, ast.Call) and isinstance(
                        env.value.func, ast.Name
                    ):
                        source = env.value.func.id
                    elif (
                        isinstance(env.value, ast.Name)
                        and env.value.id in stripped_locals
                    ):
                        source = "agent_subprocess_env"
                if source not in stripping:
                    offenders.append(
                        f"{rel}:{node.lineno} in {func.name}() spawns with "
                        f"env={'(missing)' if env is None else 'something else'}"
                    )

    assert checked >= 5, f"the AST sweep found only {checked} spawns; it has drifted"
    assert not offenders, (
        "an agent subprocess inherits the scufris credentials:\n  "
        + "\n  ".join(offenders)
        + "\nPass env=agent_subprocess_env(settings), or add the call site to "
        "`exempt` with the reason it is not the model's shell."
    )
