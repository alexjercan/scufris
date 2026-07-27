"""Tests for the agent backend.

The unit tests exercise the pure helpers (MCP overrides, steering); the
integration tests point ``codex_bin`` at a tiny fake `codex app-server` script,
so the JSON-RPC subprocess plumbing runs for real without the actual codex binary
or network.
"""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from scufris.agent import (
    AgentUnavailable,
    StreamDone,
    StreamError,
    StreamReasoningDelta,
    StreamTextDelta,
    StreamTool,
    _appserver_event,
    _git_writable_roots,
    _mcp_overrides,
    _sandbox_overrides,
    _steer,
    _stream_app_server,
    scufris_mcp_servers,
)
from scufris.config import Settings
from scufris.sessions import (
    AGENT_STEERING_PREAMBLE,
    STEERING_PREAMBLE,
    strip_steering,
)


def _enabled(*, codex_bin: str | None = None, agent_model: str = "gpt-5.5") -> Settings:
    # `_env_file=None` ignores the repo `.env`: these tests use `_enabled()` as the
    # "nothing set" baseline and assert fields are absent/defaulted (e.g. no
    # SCUFRIS_DEN_PATH / SCUFRIS_DISABLED_TOOLS in the MCP env). A dev box whose
    # `.env` sets such a knob (legit local config) would otherwise redden them while
    # `nix flake check` (no `.env` in the sandbox) stays green - the
    # isolate-tests-that-assert-config trap. Keep the baseline hermetic.
    # `_env_file` is a pydantic-settings init arg (disables the .env source); it is
    # not in the generated model signature, so mypy needs the ignore.
    return Settings(
        agent_enabled=True,
        codex_bin=codex_bin,
        agent_model=agent_model,
        _env_file=None,  # type: ignore[call-arg]
    )


def _den(path: str = "/home/op/the-den") -> Settings:
    return Settings(agent_enabled=True, den_path=Path(path), _env_file=None)  # type: ignore[call-arg]


def test_mcp_overrides_registers_scufris_for_orchestrator() -> None:
    # No den configured -> the orchestrator turn registers ONLY the scufris server.
    args = _mcp_overrides(_enabled(), is_orchestrator=True)
    joined = " ".join(args)
    assert "mcp_servers.scufris.command=" in joined
    assert 'mcp_servers.scufris.default_tools_approval_mode="approve"' in args
    assert 'approval_policy="never"' in args
    # The role env is retired (the audience split is physical); no self-agent-id.
    assert "SCUFRIS_AGENT_ROLE" not in joined
    assert "SCUFRIS_AGENT_ID" not in joined
    # den absent (den_path unset), and never the sub-agent callback server.
    assert "mcp_servers.den" not in joined
    assert "mcp_servers.agent" not in joined


def test_orchestrator_registers_scufris_and_den() -> None:
    # With a den configured, the orchestrator turn registers BOTH the scufris
    # agentic server and the den life server, each at its own module.
    joined = " ".join(_mcp_overrides(_den(), is_orchestrator=True))
    assert 'mcp_servers.scufris.args=["-m", "scufris.mcp_server"]' in joined
    assert 'mcp_servers.den.args=["-m", "scufris.den_mcp_server"]' in joined
    # Only the den server carries the den path (isolation).
    assert "mcp_servers.den.env.SCUFRIS_DEN_PATH=" in joined
    assert "mcp_servers.scufris.env.SCUFRIS_DEN_PATH" not in joined
    assert "/home/op/the-den" in joined
    # No callback server on an orchestrator turn.
    assert "mcp_servers.agent" not in joined


def test_subagent_registers_only_callback_server() -> None:
    # A regular sub-agent turn registers ONLY the `agent` callback server (id
    # `agent`, request_input/report_back), carrying its own id so the callbacks can
    # address it, plus the API base to POST back. It never gets scufris/den.
    joined = " ".join(_mcp_overrides(_den(), is_orchestrator=False, agent_id="builder"))
    assert 'mcp_servers.agent.args=["-m", "scufris.agent_mcp_server"]' in joined
    assert 'mcp_servers.agent.env.SCUFRIS_AGENT_ID="builder"' in joined
    assert "mcp_servers.agent.env.SCUFRIS_API_BASE=" in joined
    assert "mcp_servers.scufris" not in joined
    assert "mcp_servers.den" not in joined
    # A regular turn WITHOUT an id gets no scufris server at all.
    assert "mcp_servers." not in " ".join(_mcp_overrides(_den())).replace(
        'approval_policy="never"', ""
    )


def test_mcp_overrides_orchestrator_wins_over_agent_id() -> None:
    # is_orchestrator wins: the orchestrator servers are registered, never the
    # sub-agent callback server, even if an agent_id is also passed.
    joined = " ".join(
        _mcp_overrides(_enabled(), is_orchestrator=True, agent_id="orchestrator")
    )
    assert "mcp_servers.scufris.command=" in joined
    assert "mcp_servers.agent" not in joined


def test_mcp_overrides_empty_when_tools_disabled() -> None:
    settings = Settings(agent_enabled=True, agent_tools_enabled=False)
    assert _mcp_overrides(settings, is_orchestrator=True) == []
    assert _mcp_overrides(settings, is_orchestrator=False, agent_id="builder") == []


def test_mcp_overrides_passes_disabled_tools_env_to_both_orch_servers() -> None:
    settings = Settings(
        agent_enabled=True,
        den_path=Path("/home/op/the-den"),
        disabled_tools=["list_processes", "macros_add_food"],
        _env_file=None,  # type: ignore[call-arg]
    )
    joined = " ".join(_mcp_overrides(settings, is_orchestrator=True))
    assert "mcp_servers.scufris.env.SCUFRIS_DISABLED_TOOLS=" in joined
    assert "mcp_servers.den.env.SCUFRIS_DISABLED_TOOLS=" in joined
    assert "list_processes,macros_add_food" in joined


def test_mcp_overrides_no_disabled_env_when_none() -> None:
    joined = " ".join(_mcp_overrides(_enabled(), is_orchestrator=True))
    assert "SCUFRIS_DISABLED_TOOLS" not in joined


def test_scufris_mcp_servers_orchestrator() -> None:
    # No den -> just the scufris server; with den -> scufris + den.
    plain = scufris_mcp_servers(_enabled(), is_orchestrator=True)
    assert [s.server_id for s in plain] == ["scufris"]
    assert list(plain[0].args) == ["-m", "scufris.mcp_server"]
    assert plain[0].env["SCUFRIS_API_BASE"].startswith("http://")
    assert "SCUFRIS_AGENT_ID" not in plain[0].env
    withden = scufris_mcp_servers(_den(), is_orchestrator=True)
    assert [s.server_id for s in withden] == ["scufris", "den"]
    den = next(s for s in withden if s.server_id == "den")
    assert list(den.args) == ["-m", "scufris.den_mcp_server"]
    assert den.env["SCUFRIS_DEN_PATH"] == "/home/op/the-den"


def test_scufris_mcp_servers_agent() -> None:
    servers = scufris_mcp_servers(_enabled(), agent_id="builder")
    assert [s.server_id for s in servers] == ["agent"]
    assert list(servers[0].args) == ["-m", "scufris.agent_mcp_server"]
    assert servers[0].env["SCUFRIS_AGENT_ID"] == "builder"
    # A den never rides a sub-agent turn, even when configured.
    assert [s.server_id for s in scufris_mcp_servers(_den(), agent_id="b")] == ["agent"]


def test_orchestrator_scufris_env_has_session_id() -> None:
    resumed = scufris_mcp_servers(
        _enabled(), is_orchestrator=True, orch_session_id="chat-1"
    )
    assert resumed[0].env["SCUFRIS_ORCH_SESSION_ID"] == "chat-1"
    fresh = scufris_mcp_servers(_enabled(), is_orchestrator=True, orch_session_id="")
    assert "SCUFRIS_ORCH_SESSION_ID" not in fresh[0].env
    # A sub-agent's callback server never carries it (it cannot spawn).
    agent = scufris_mcp_servers(
        _enabled(), agent_id="builder", orch_session_id="chat-1"
    )
    assert "SCUFRIS_ORCH_SESSION_ID" not in agent[0].env


def test_scufris_mcp_servers_empty_without_id_or_when_disabled() -> None:
    assert scufris_mcp_servers(_enabled()) == []
    assert scufris_mcp_servers(_enabled(), is_orchestrator=False) == []
    off = Settings(agent_tools_enabled=False)
    assert scufris_mcp_servers(off, is_orchestrator=True) == []
    assert scufris_mcp_servers(off, agent_id="builder") == []


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


def test_steer_orchestrator_gets_journal_food_chain() -> None:
    # A bare "log that I had 2 eggs" (no tool names) must reach the food tools on its
    # own: codex only honors tool-choice steering that rides the turn prompt, so the
    # den-journal clause carries the meal chain macros_lookup -> journal_add_macros.
    steered = _steer(_enabled(), "log that I had 2 eggs", is_orchestrator=True)
    assert "macros_lookup" in steered
    assert "journal_add_macros" in steered
    # General scope: the wider den-journal write surface is pointed at too, so
    # "add a task" / "log 80kg" / "check off gym" land on the tools as well.
    assert "journal_add_task" in steered
    assert "journal_log_weight" in steered
    # Orchestrator-only: a sub-agent turn never sees the journal steering (it has
    # none of these tools - the den server is never registered on a sub-agent turn).
    sub = _steer(_enabled(), "do the task", agent_id="a-1")
    assert "macros_lookup" not in sub
    assert "journal_add_macros" not in sub
    # Still one strippable block, so titles/transcripts stay clean.
    assert strip_steering(steered) == "log that I had 2 eggs"


def test_steer_orchestrator_gets_agent_delegation_chain() -> None:
    # "implement task X using codex" (no tool names) must make the orchestrator
    # create-and-run an agent on its own: the delegation chain rides the turn prompt
    # (create_agent then run_agent), since codex ignores tool docstrings for choice.
    steered = _steer(
        _enabled(), "implement task 20260724-012212 using codex", is_orchestrator=True
    )
    assert "create_agent" in steered
    assert "run_agent" in steered
    # The project lookup and follow-up tools are named too.
    assert "list_projects" in steered
    assert "agent_status" in steered
    # Orchestrator-only: a sub-agent turn never sees the delegation tools (it cannot
    # create or run agents - the scufris server is never registered on its turn).
    sub = _steer(_enabled(), "do the task", agent_id="a-1")
    assert "create_agent" not in sub
    assert "run_agent" not in sub
    # Still one strippable block, so titles/transcripts stay clean.
    assert strip_steering(steered) == "implement task 20260724-012212 using codex"


def test_steer_agent_told_to_implement_the_task_end_to_end() -> None:
    # A spawned sub-agent must know its job is to CARRY THE TASK TO COMPLETION, not
    # narrate a plan and stop (the reported 1-turn, 0-tool-call failure). The work
    # clause is backend-agnostic: it does not depend on the flow skill (codex has
    # none), and it keeps the request_input-when-blocked instruction.
    steered = _steer(_enabled(), "implement the task", agent_id="a-1")
    lowered = steered.lower()
    assert "request_input" in steered  # still signals when blocked
    assert "report_back" in steered  # and reports its result when finished
    assert "end-to-end" in lowered or "to completion" in lowered
    # It does not steer the sub-agent to the orchestrator-only delegation tools.
    assert "create_agent" not in steered
    assert "run_agent" not in steered
    # One strippable block, so titles/transcripts stay clean.
    assert strip_steering(steered) == "implement the task"


def test_agent_steering_stays_a_single_block() -> None:
    # Same single-block invariant as the orchestrator preamble: the sub-agent
    # preamble carries request_input + the work clause in ONE [scufris-tools] block,
    # because strip_steering removes only the first leading block (count=1).
    assert AGENT_STEERING_PREAMBLE.count("[scufris-tools]") == 1
    assert AGENT_STEERING_PREAMBLE.count("[/scufris-tools]") == 1


def test_orchestrator_steering_stays_a_single_block() -> None:
    # Ledger invariant (orchestrator-steering-is-one-block-two-clauses): every
    # orchestrator clause lives in ONE [scufris-tools] block, because strip_steering
    # removes only the FIRST leading block (regex count=1). A second sentinel block
    # would survive uncleaned in titles/transcripts. Adding the journal clause must
    # not break this.
    assert STEERING_PREAMBLE.count("[scufris-tools]") == 1
    assert STEERING_PREAMBLE.count("[/scufris-tools]") == 1
    # report_back is a SUB-AGENT callback; the orchestrator receives it via
    # pending_agents, so its steering preamble does not name report_back.
    assert "report_back" not in STEERING_PREAMBLE


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


# A fake app-server that streams deltas SLOWLY: five deltas 0.15s apart, so the
# turn's total wall-clock (~0.75s) exceeds a small `agent_timeout_seconds` while
# no single gap between lines does. Drives the idle-timeout regression: the old
# per-turn wall-clock deadline killed this mid-stream; an idle guard lets it run.
_FAKE_APPSERVER_SLOW = """#!/usr/bin/env python3
import sys, json, time
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
        for i in range(5):
            time.sleep(0.15)
            out({"method": "item/agentMessage/delta", "params": {"delta": str(i)}})
        out({"method": "turn/completed", "params": {}})
        break
"""


# A fake app-server that goes SILENT after setup: it acks turn/start then emits
# nothing (sleeps well past any idle bound). The idle guard must still cut this
# as a genuine stall.
_FAKE_APPSERVER_STALL = """#!/usr/bin/env python3
import sys, json, time
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
        time.sleep(5)
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


_HAS_GIT = shutil.which("git") is not None
_needs_git = pytest.mark.skipif(not _HAS_GIT, reason="git not on PATH")


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for cmd in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@t.t"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(cmd, cwd=path, check=True, capture_output=True)


@_needs_git
def test_git_writable_roots_plain_repo(tmp_path: Path) -> None:
    """A plain repo yields exactly its `.git` dir (git-dir == git-common-dir)."""
    _init_repo(tmp_path)
    roots = _git_writable_roots(str(tmp_path))
    assert roots == [str((tmp_path / ".git").resolve())]


@_needs_git
def test_git_writable_roots_worktree(tmp_path: Path) -> None:
    """A sprout-style worktree yields TWO roots: its own git dir under
    `.git/worktrees/<name>` AND the parent repo's shared common `.git`. Both are
    needed - a commit from the worktree writes both."""
    main = tmp_path / "main"
    _init_repo(main)
    (main / "f.txt").write_text("hi")
    subprocess.run(["git", "add", "."], cwd=main, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-qm", "init"], cwd=main, check=True, capture_output=True
    )
    wt = tmp_path / "wt"
    subprocess.run(
        ["git", "worktree", "add", "-q", str(wt)],
        cwd=main,
        check=True,
        capture_output=True,
    )
    roots = _git_writable_roots(str(wt))
    common = str((main / ".git").resolve())
    gitdir = str((main / ".git" / "worktrees" / "wt").resolve())
    assert set(roots) == {gitdir, common}


@_needs_git
def test_git_writable_roots_non_repo(tmp_path: Path) -> None:
    """A directory that is not a git repo re-grants nothing."""
    assert _git_writable_roots(str(tmp_path)) == []


def test_git_writable_roots_no_cwd() -> None:
    assert _git_writable_roots(None) == []


@_needs_git
def test_sandbox_overrides_only_for_workspace_write(tmp_path: Path) -> None:
    """`edit` (workspace-write) re-grants git dirs; `manual`/`auto` do not - only
    workspace-write protects `.git`, so the override is meaningless elsewhere."""
    _init_repo(tmp_path)
    cwd = str(tmp_path)
    edit = _sandbox_overrides("workspace-write", cwd)
    assert edit[0] == "-c"
    assert edit[1].startswith("sandbox_workspace_write.writable_roots=")
    assert str((tmp_path / ".git").resolve()) in edit[1]
    assert _sandbox_overrides("read-only", cwd) == []
    assert _sandbox_overrides("danger-full-access", cwd) == []


@_needs_git
async def test_stream_app_server_edit_grants_git_on_argv(tmp_path: Path) -> None:
    """End-to-end: an `edit` (workspace-write) turn in a git repo puts the
    writable_roots override on the codex argv, so the agent can commit; a
    `danger-full-access` turn does not (it already has full access)."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    fake = _write_fake_appserver(tmp_path / "codex", body=_FAKE_APPSERVER_ARGV)
    settings = Settings(agent_enabled=True, codex_bin=fake, agent_model="")

    async def argv_for(sandbox: str) -> str:
        _ = [
            e
            async for e in _stream_app_server(
                settings, "hi", cwd=str(repo), sandbox=sandbox
            )
        ]
        return " ".join(json.loads((repo / "argv.json").read_text()))

    edit_argv = await argv_for("workspace-write")
    assert "sandbox_workspace_write.writable_roots=" in edit_argv
    assert str((repo / ".git").resolve()) in edit_argv

    auto_argv = await argv_for("danger-full-access")
    assert "sandbox_workspace_write.writable_roots=" not in auto_argv


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


async def test_stream_app_server_slow_but_streaming_completes(tmp_path: Path) -> None:
    """A turn whose TOTAL time exceeds `agent_timeout_seconds` but that never goes
    silent longer than it must complete with all its events. The old per-turn
    wall-clock deadline killed this mid-stream ("app-server timed out"); the idle
    guard (timeout resets on each streamed line) lets it run to turn/completed."""
    fake = _write_fake_appserver(tmp_path / "codex", body=_FAKE_APPSERVER_SLOW)
    # Idle bound 0.4s: comfortably above the 0.15s inter-delta gap, well below the
    # ~0.75s total. A wall-clock deadline of 0.4s would fire mid-stream.
    settings = Settings(
        agent_enabled=True,
        codex_bin=fake,
        agent_model="",
        agent_tools_enabled=False,
        agent_timeout_seconds=0.4,
    )
    events = [e async for e in _stream_app_server(settings, "hi")]

    assert not any(isinstance(e, StreamError) for e in events)
    deltas = [e.delta for e in events if isinstance(e, StreamTextDelta)]
    assert deltas == ["0", "1", "2", "3", "4"]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.reply.text == "01234"


async def test_stream_app_server_idle_stall_times_out(tmp_path: Path) -> None:
    """The idle guard still cuts a GENUINE stall: an app-server that acks the turn
    then emits nothing yields a timeout StreamError once the idle bound elapses."""
    fake = _write_fake_appserver(tmp_path / "codex", body=_FAKE_APPSERVER_STALL)
    settings = Settings(
        agent_enabled=True,
        codex_bin=fake,
        agent_model="",
        agent_tools_enabled=False,
        agent_timeout_seconds=0.3,
    )
    events = [e async for e in _stream_app_server(settings, "hi")]

    assert events, "expected at least a StreamError"
    last = events[-1]
    assert isinstance(last, StreamError)
    assert "timed out" in last.detail


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
