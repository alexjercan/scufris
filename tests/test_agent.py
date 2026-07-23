"""Tests for the agent backend.

The unit tests exercise the pure helpers (MCP overrides, steering); the
integration tests point ``codex_bin`` at a tiny fake `codex app-server` script,
so the JSON-RPC subprocess plumbing runs for real without the actual codex binary
or network.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from scufris.agent import (
    AgentUnavailable,
    StreamDone,
    StreamReasoningDelta,
    StreamTextDelta,
    StreamTool,
    _appserver_event,
    _mcp_overrides,
    _steer,
    _stream_app_server,
    scufris_mcp_server,
)
from scufris.config import McpServerSpec, Settings
from scufris.sessions import (
    AGENT_STEERING_PREAMBLE,
    STEERING_PREAMBLE,
    strip_steering,
)


def _enabled(*, codex_bin: str | None = None, agent_model: str = "gpt-5.5") -> Settings:
    return Settings(agent_enabled=True, codex_bin=codex_bin, agent_model=agent_model)


def test_mcp_overrides_registers_scufris_for_orchestrator() -> None:
    args = _mcp_overrides(_enabled(), is_orchestrator=True)
    joined = " ".join(args)
    assert "mcp_servers.scufris.command=" in joined
    assert "mcp_servers.scufris.args=" in joined
    assert 'mcp_servers.scufris.default_tools_approval_mode="approve"' in args
    assert 'approval_policy="never"' in args
    # The orchestrator role; no self-agent-id (it addresses others explicitly).
    assert 'mcp_servers.scufris.env.SCUFRIS_AGENT_ROLE="orchestrator"' in args
    assert "SCUFRIS_AGENT_ID" not in joined


def test_mcp_overrides_agent_role_for_a_regular_agent() -> None:
    """A regular agent gets the scufris server in the AGENT role (only the
    request_input callback), carrying its own id so the callback can address it
    (BC2). Without an agent_id it still gets none."""
    settings = _enabled()
    agent = " ".join(
        _mcp_overrides(settings, is_orchestrator=False, agent_id="builder")
    )
    assert "mcp_servers.scufris.command=" in agent
    assert 'mcp_servers.scufris.env.SCUFRIS_AGENT_ROLE="agent"' in agent
    assert 'mcp_servers.scufris.env.SCUFRIS_AGENT_ID="builder"' in agent
    # A regular agent WITHOUT an id (or a caller that forgets it) gets no scufris.
    assert "mcp_servers.scufris" not in " ".join(
        _mcp_overrides(settings, is_orchestrator=False)
    )
    assert "mcp_servers.scufris" not in " ".join(_mcp_overrides(settings))


def test_mcp_overrides_orchestrator_wins_over_agent_id() -> None:
    """The orchestrator role takes precedence: it never gets the agent role even
    if an agent_id is also passed."""
    args = " ".join(
        _mcp_overrides(_enabled(), is_orchestrator=True, agent_id="orchestrator")
    )
    assert 'env.SCUFRIS_AGENT_ROLE="orchestrator"' in args
    assert 'env.SCUFRIS_AGENT_ROLE="agent"' not in args


def test_mcp_overrides_empty_when_tools_disabled() -> None:
    settings = Settings(agent_enabled=True, agent_tools_enabled=False)
    assert _mcp_overrides(settings, is_orchestrator=True) == []
    assert _mcp_overrides(settings, is_orchestrator=False, agent_id="builder") == []


def test_mcp_overrides_appends_configured_servers_for_any_agent() -> None:
    # Operator-declared servers are global config and reach EVERY agent; only the
    # built-in scufris server is orchestrator-scoped.
    settings = Settings(
        agent_enabled=True,
        mcp_servers=[
            McpServerSpec(
                id="fs", command="mcp-fs", args=["--root", "/tmp"], approve=False
            )
        ],
    )
    joined = " ".join(_mcp_overrides(settings, is_orchestrator=False))
    assert 'mcp_servers.fs.command="mcp-fs"' in joined
    assert "mcp_servers.fs.args=" in joined
    # approve=False -> no auto-approval line for this server
    assert "mcp_servers.fs.default_tools_approval_mode" not in joined
    # ...and still no scufris for the regular agent.
    assert "mcp_servers.scufris" not in joined


def test_mcp_overrides_skips_invalid_or_reserved_id() -> None:
    settings = Settings(
        agent_enabled=True,
        mcp_servers=[
            McpServerSpec(id="bad.id", command="x"),
            McpServerSpec(id="scufris", command="evil"),
        ],
    )
    joined = " ".join(_mcp_overrides(settings, is_orchestrator=True))
    assert "bad.id" not in joined  # invalid id skipped
    assert "evil" not in joined  # reserved scufris id not overridden


def test_mcp_overrides_passes_disabled_tools_env() -> None:
    settings = Settings(
        agent_enabled=True, disabled_tools=["list_processes", "disk_usage"]
    )
    joined = " ".join(_mcp_overrides(settings, is_orchestrator=True))
    assert "mcp_servers.scufris.env.SCUFRIS_DISABLED_TOOLS=" in joined
    assert "list_processes,disk_usage" in joined


def test_mcp_overrides_no_disabled_env_when_none() -> None:
    joined = " ".join(_mcp_overrides(_enabled(), is_orchestrator=True))
    assert "SCUFRIS_DISABLED_TOOLS" not in joined


def test_mcp_overrides_injects_api_base_for_orchestrator() -> None:
    # The orchestrator server carries the dashboard's API base so its control tools
    # can call back over HTTP; a regular agent has no scufris server at all.
    settings = Settings(agent_enabled=True, host="127.0.0.1", port=8123)
    joined = " ".join(_mcp_overrides(settings, is_orchestrator=True))
    assert "mcp_servers.scufris.env.SCUFRIS_API_BASE=" in joined
    assert "http://127.0.0.1:8123" in joined
    assert "SCUFRIS_API_BASE" not in " ".join(
        _mcp_overrides(settings, is_orchestrator=False)
    )


def test_scufris_mcp_server_orchestrator_role() -> None:
    """The shared core carries the orchestrator role env + API base; codex and
    claude both format THIS, so they cannot drift on what a role exposes."""
    server = scufris_mcp_server(
        Settings(agent_enabled=True, host="127.0.0.1", port=8123),
        is_orchestrator=True,
    )
    assert server is not None
    assert list(server.args) == ["-m", "scufris.mcp_server"]
    assert server.env["SCUFRIS_AGENT_ROLE"] == "orchestrator"
    assert server.env["SCUFRIS_API_BASE"] == "http://127.0.0.1:8123"
    # The orchestrator addresses others explicitly; it has no self-id.
    assert "SCUFRIS_AGENT_ID" not in server.env


def test_scufris_mcp_server_agent_role_threads_id() -> None:
    server = scufris_mcp_server(_enabled(), agent_id="builder")
    assert server is not None
    assert server.env["SCUFRIS_AGENT_ROLE"] == "agent"
    assert server.env["SCUFRIS_AGENT_ID"] == "builder"


def test_scufris_mcp_server_disabled_tools_passthrough() -> None:
    server = scufris_mcp_server(
        Settings(agent_enabled=True, disabled_tools=["list_processes", "disk_usage"]),
        is_orchestrator=True,
    )
    assert server is not None
    assert server.env["SCUFRIS_DISABLED_TOOLS"] == "list_processes,disk_usage"
    # No disabled set -> no env key.
    plain = scufris_mcp_server(_enabled(), is_orchestrator=True)
    assert plain is not None
    assert "SCUFRIS_DISABLED_TOOLS" not in plain.env


def test_scufris_mcp_server_none_without_role_or_when_disabled() -> None:
    # A regular agent with no id has nothing to address the callback back to.
    assert scufris_mcp_server(_enabled()) is None
    assert scufris_mcp_server(_enabled(), is_orchestrator=False) is None
    # Tools disabled -> no scufris server for either role.
    off = Settings(agent_tools_enabled=False)
    assert scufris_mcp_server(off, is_orchestrator=True) is None
    assert scufris_mcp_server(off, agent_id="builder") is None


def test_steer_prepends_preamble_for_orchestrator() -> None:
    steered = _steer(_enabled(), "tell me about this host", is_orchestrator=True)
    assert steered.startswith(STEERING_PREAMBLE)
    assert steered.endswith("tell me about this host")
    # The preamble is transparently removable, so titles/transcripts stay clean.
    assert strip_steering(steered) == "tell me about this host"


def test_steer_orchestrator_gets_comms_protocol() -> None:
    # The orchestrator is steered to close the request_input round-trip on the poll
    # path: find blocked sub-agents, answer them, and clear them.
    steered = _steer(_enabled(), "check on my agents", is_orchestrator=True)
    assert "pending_agents" in steered
    assert "message_agent" in steered
    assert "acknowledge" in steered
    # A sub-agent turn must NOT see the orchestrator-only comms protocol (SC1 owns
    # the sub-agent's steering, and it has none of these tools).
    sub = _steer(_enabled(), "do the task", agent_id="a-1")
    assert "pending_agents" not in sub
    assert "message_agent" not in sub
    # Still one strippable block, so titles/transcripts stay clean.
    assert strip_steering(steered) == "check on my agents"


def test_steer_agent_gets_request_input_preamble() -> None:
    # A tool-having codex sub-agent (agent_id set, tools on, not orchestrator) is
    # told to signal via request_input when blocked, and gets the AGENT preamble -
    # not the orchestrator's host-tools one.
    steered = _steer(_enabled(), "do the task", agent_id="a-1")
    assert steered.startswith(AGENT_STEERING_PREAMBLE)
    assert steered.endswith("do the task")
    assert "request_input" in steered
    assert STEERING_PREAMBLE not in steered
    # Same sentinel markers, so strip_steering cleans it out of titles/transcripts.
    assert strip_steering(steered) == "do the task"


def test_steer_orchestrator_ignores_agent_id() -> None:
    # The orchestrator keeps the host-tools preamble even if an id is passed; it is
    # never handed the sub-agent's request_input steering.
    steered = _steer(_enabled(), "hi", is_orchestrator=True, agent_id="a-1")
    assert steered.startswith(STEERING_PREAMBLE)
    assert AGENT_STEERING_PREAMBLE not in steered


def test_steer_noop_for_toolless_agent() -> None:
    # A claude sub-agent has no scufris server (no agent_id), so no steering rides.
    assert _steer(_enabled(), "hello", is_orchestrator=False) == "hello"
    assert _steer(_enabled(), "hello") == "hello"  # default is not-orchestrator


def test_steer_noop_when_tools_disabled() -> None:
    # Tools off wins over every role, including a would-be tool-having sub-agent.
    settings = Settings(agent_enabled=True, agent_tools_enabled=False)
    assert _steer(settings, "hello", is_orchestrator=True) == "hello"
    assert _steer(settings, "hello", agent_id="a-1") == "hello"


# --- app-server backend ---


def test_appserver_event_maps_text_and_reasoning_deltas() -> None:
    text = _appserver_event(
        {"method": "item/agentMessage/delta", "params": {"delta": "Hel"}}
    )
    assert isinstance(text, StreamTextDelta) and text.delta == "Hel"
    reason = _appserver_event(
        {"method": "item/reasoning/textDelta", "params": {"delta": "hmm"}}
    )
    assert isinstance(reason, StreamReasoningDelta) and reason.delta == "hmm"


def test_appserver_event_maps_tool_and_ignores_others() -> None:
    tool = _appserver_event(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "mcpToolCall",
                    "tool": "host_stats",
                    "status": "completed",
                }
            },
        }
    )
    assert isinstance(tool, StreamTool) and tool.tool.tool == "host_stats"
    assert _appserver_event({"method": "turn/started", "params": {}}) is None


_FAKE_APPSERVER = """#!/usr/bin/env python3
import sys, json
def out(o):
    sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line); rid = req.get("id"); m = req.get("method")
    if m == "initialize":
        out({"id": rid, "result": {}})
    elif m in ("thread/start", "thread/resume"):
        out({"id": rid, "result": {"thread": {"id": "t-1"}}})
    elif m == "turn/start":
        out({"id": rid, "result": {"turn": {}}})
        out({"method": "item/agentMessage/delta", "params": {"delta": "Hel"}})
        out({"method": "item/agentMessage/delta", "params": {"delta": "lo"}})
        out({"method": "thread/tokenUsage/updated",
             "params": {"tokenUsage": {"total": {"inputTokens": 5, "outputTokens": 2}}}})
        out({"method": "turn/completed", "params": {}})
        break
"""


# A fake app-server that LOGS each received request (method + params) to
# `reqs.jsonl` in its cwd, so a test can assert what scufris sent.
_FAKE_APPSERVER_LOG = """#!/usr/bin/env python3
import sys, json, os
log = os.path.join(os.getcwd(), "reqs.jsonl")
def out(o):
    sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line); rid = req.get("id"); m = req.get("method")
    with open(log, "a") as f:
        f.write(json.dumps({"method": m, "params": req.get("params")}) + "\\n")
    if m == "initialize":
        out({"id": rid, "result": {}})
    elif m in ("thread/start", "thread/resume"):
        out({"id": rid, "result": {"thread": {"id": "t-1"}}})
    elif m == "turn/start":
        out({"id": rid, "result": {"turn": {}}})
        out({"method": "turn/completed", "params": {}})
        break
"""


# A fake app-server that dumps its own argv to `argv.json` in its cwd, so a test
# can assert which `-c mcp_servers...` overrides scufris put on the command line.
_FAKE_APPSERVER_ARGV = """#!/usr/bin/env python3
import sys, json, os
with open(os.path.join(os.getcwd(), "argv.json"), "w") as f:
    json.dump(sys.argv, f)
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line); rid = req.get("id"); m = req.get("method")
    if m == "initialize":
        out = {"id": rid, "result": {}}
    elif m in ("thread/start", "thread/resume"):
        out = {"id": rid, "result": {"thread": {"id": "t-1"}}}
    elif m == "turn/start":
        sys.stdout.write(json.dumps({"id": rid, "result": {"turn": {}}}) + "\\n")
        sys.stdout.write(json.dumps({"method": "turn/completed", "params": {}}) + "\\n")
        sys.stdout.flush()
        break
    else:
        continue
    sys.stdout.write(json.dumps(out) + "\\n"); sys.stdout.flush()
"""


async def _argv_of_turn(tmp_path: Path, *, is_orchestrator: bool) -> list[str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    fake = _write_fake_appserver(tmp_path / "codex", body=_FAKE_APPSERVER_ARGV)
    settings = Settings(agent_enabled=True, codex_bin=fake, agent_model="")
    _ = [
        e
        async for e in _stream_app_server(
            settings, "hi", cwd=str(tmp_path), is_orchestrator=is_orchestrator
        )
    ]
    return json.loads((tmp_path / "argv.json").read_text())


async def test_stream_app_server_scufris_argv_scoped_to_orchestrator(
    tmp_path: Path,
) -> None:
    """End-to-end through the spawn: the orchestrator turn's codex argv carries the
    scufris MCP server; a regular agent turn's does not. Proves is_orchestrator is
    threaded all the way from `stream` to the process command line."""
    orch = " ".join(await _argv_of_turn(tmp_path / "o", is_orchestrator=True))
    regular = " ".join(await _argv_of_turn(tmp_path / "r", is_orchestrator=False))
    assert "mcp_servers.scufris.command=" in orch
    assert "mcp_servers.scufris" not in regular


async def test_stream_app_server_resume_re_sends_sandbox(tmp_path: Path) -> None:
    """thread/resume MUST carry the sandbox: each turn is a fresh app-server
    process and a resumed thread does not restore its start sandbox, so without
    this an auto/edit agent reverts to read-only after turn 1 (20260721-183828)."""
    fake = tmp_path / "codex"
    fake.write_text(
        _FAKE_APPSERVER_LOG.replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1)
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    settings = Settings(
        agent_enabled=True,
        codex_bin=str(fake),
        agent_model="",
        agent_tools_enabled=False,
    )
    # Resume an existing thread with a WRITABLE sandbox.
    _ = [
        e
        async for e in _stream_app_server(
            settings,
            "hi",
            "t-existing",
            cwd=str(tmp_path),
            sandbox="workspace-write",
        )
    ]
    reqs = [
        json.loads(line) for line in (tmp_path / "reqs.jsonl").read_text().splitlines()
    ]
    resume = next(r for r in reqs if r["method"] == "thread/resume")
    assert resume["params"]["threadId"] == "t-existing"
    assert resume["params"]["sandbox"] == "workspace-write"  # the fix


async def test_stream_app_server_start_sends_sandbox(tmp_path: Path) -> None:
    """A new thread also carries the sandbox (turn 1 was already correct)."""
    fake = tmp_path / "codex"
    fake.write_text(
        _FAKE_APPSERVER_LOG.replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1)
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    settings = Settings(
        agent_enabled=True,
        codex_bin=str(fake),
        agent_model="",
        agent_tools_enabled=False,
    )
    _ = [
        e
        async for e in _stream_app_server(
            settings, "hi", None, cwd=str(tmp_path), sandbox="danger-full-access"
        )
    ]
    reqs = [
        json.loads(line) for line in (tmp_path / "reqs.jsonl").read_text().splitlines()
    ]
    start = next(r for r in reqs if r["method"] == "thread/start")
    assert start["params"]["sandbox"] == "danger-full-access"


async def test_stream_app_server_streams_text_deltas(tmp_path: Path) -> None:
    fake = tmp_path / "codex"
    # Resolve python3 (sys.executable) instead of `/usr/bin/env python3`, absent
    # in the nix check sandbox.
    fake.write_text(
        _FAKE_APPSERVER.replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1)
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    settings = Settings(
        agent_enabled=True,
        codex_bin=str(fake),
        agent_model="",
        agent_tools_enabled=False,
    )
    events = [e async for e in _stream_app_server(settings, "hi")]

    deltas = [e.delta for e in events if isinstance(e, StreamTextDelta)]
    assert deltas == ["Hel", "lo"]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.reply.text == "Hello"
    assert done.session_id == "t-1"
    assert done.reply.usage is not None
    assert done.reply.usage.input_tokens == 5


def _write_fake_appserver(path: Path, body: str = _FAKE_APPSERVER_LOG) -> str:
    path.write_text(body.replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1))
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


async def test_stream_app_server_missing_binary_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared `_resolve_codex_bin` guard: no codex on PATH -> AgentUnavailable
    surfaces when the stream is driven (it raises inside the generator body)."""
    monkeypatch.setattr("scufris.agent.shutil.which", lambda _name: None)
    settings = Settings(agent_enabled=True, codex_bin=None, agent_tools_enabled=False)
    with pytest.raises(AgentUnavailable, match="codex CLI not found"):
        _ = [e async for e in _stream_app_server(settings, "hi")]


async def test_stream_app_server_runs_in_the_given_cwd(tmp_path: Path) -> None:
    """The cwd seam: a turn's subprocess runs in the supplied project dir, not the
    server's cwd - the foundation for per-agent, per-project runs. The fake logs to
    `reqs.jsonl` in its own cwd, so its presence in workdir proves the cwd took."""
    workdir = tmp_path / "project"
    workdir.mkdir()
    fake = _write_fake_appserver(tmp_path / "codex")
    settings = Settings(
        agent_enabled=True, codex_bin=fake, agent_model="", agent_tools_enabled=False
    )
    _ = [e async for e in _stream_app_server(settings, "hi", cwd=str(workdir))]
    assert (workdir / "reqs.jsonl").exists()
    assert not (tmp_path / "reqs.jsonl").exists()


async def test_stream_app_server_attaches_images(tmp_path: Path) -> None:
    """Attached images ride the turn as `localImage` items alongside the text."""
    fake = _write_fake_appserver(tmp_path / "codex")
    settings = Settings(
        agent_enabled=True, codex_bin=fake, agent_model="", agent_tools_enabled=False
    )
    _ = [
        e
        async for e in _stream_app_server(
            settings, "look", cwd=str(tmp_path), image_paths=["/tmp/a.png"]
        )
    ]
    reqs = [
        json.loads(line) for line in (tmp_path / "reqs.jsonl").read_text().splitlines()
    ]
    turn = next(r for r in reqs if r["method"] == "turn/start")
    kinds = [item.get("type") for item in turn["params"]["input"]]
    assert "localImage" in kinds
    image = next(i for i in turn["params"]["input"] if i.get("type") == "localImage")
    assert image["path"] == "/tmp/a.png"
