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
)
from scufris.config import McpServerSpec, Settings
from scufris.sessions import STEERING_PREAMBLE, strip_steering


def _enabled(*, codex_bin: str | None = None, agent_model: str = "gpt-5.5") -> Settings:
    return Settings(agent_enabled=True, codex_bin=codex_bin, agent_model=agent_model)


def test_mcp_overrides_registers_scufris_for_orchestrator() -> None:
    args = _mcp_overrides(_enabled(), is_orchestrator=True)
    joined = " ".join(args)
    assert "mcp_servers.scufris.command=" in joined
    assert "mcp_servers.scufris.args=" in joined
    assert 'mcp_servers.scufris.default_tools_approval_mode="approve"' in args
    assert 'approval_policy="never"' in args


def test_mcp_overrides_scopes_scufris_to_orchestrator() -> None:
    """The built-in scufris server is registered ONLY for the orchestrator; a
    regular agent gets none (it draws its tools from its own project config)."""
    settings = _enabled()
    orch = " ".join(_mcp_overrides(settings, is_orchestrator=True))
    regular = " ".join(_mcp_overrides(settings, is_orchestrator=False))
    assert "mcp_servers.scufris" in orch
    assert "mcp_servers.scufris" not in regular
    # is_orchestrator defaults to False - a caller that forgets it gets no scufris.
    assert "mcp_servers.scufris" not in " ".join(_mcp_overrides(settings))


def test_mcp_overrides_empty_when_tools_disabled() -> None:
    settings = Settings(agent_enabled=True, agent_tools_enabled=False)
    assert _mcp_overrides(settings, is_orchestrator=True) == []


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
    settings = Settings(agent_enabled=True, disabled_tools=["tatr_new", "disk_usage"])
    joined = " ".join(_mcp_overrides(settings, is_orchestrator=True))
    assert "mcp_servers.scufris.env.SCUFRIS_DISABLED_TOOLS=" in joined
    assert "tatr_new,disk_usage" in joined


def test_mcp_overrides_no_disabled_env_when_none() -> None:
    joined = " ".join(_mcp_overrides(_enabled(), is_orchestrator=True))
    assert "SCUFRIS_DISABLED_TOOLS" not in joined


def test_steer_prepends_preamble_for_orchestrator() -> None:
    steered = _steer(_enabled(), "tell me about this host", is_orchestrator=True)
    assert steered.startswith(STEERING_PREAMBLE)
    assert steered.endswith("tell me about this host")
    # The preamble is transparently removable, so titles/transcripts stay clean.
    assert strip_steering(steered) == "tell me about this host"


def test_steer_noop_for_regular_agent() -> None:
    # A regular agent has no scufris tools, so steering toward them is meaningless.
    assert _steer(_enabled(), "hello", is_orchestrator=False) == "hello"
    assert _steer(_enabled(), "hello") == "hello"  # default is not-orchestrator


def test_steer_noop_when_tools_disabled() -> None:
    settings = Settings(agent_enabled=True, agent_tools_enabled=False)
    assert _steer(settings, "hello", is_orchestrator=True) == "hello"


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
