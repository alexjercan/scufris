"""The `host` MCP server: the host agent's toolset, and what it does NOT hold.

These cover ``mcp_host_tools`` (where the host tools are DEFINED, once) through
the ``host`` server that registers all of them - the read-only inspection half
that the orchestrator shares, and the propose-only mutating half that only this
audience has. The audience split itself is asserted from both sides here and in
``test_mcp_server.py``.

The inspection tests run against the REAL host: they are read-only, and a fake
would prove only that the fake works. The parsers themselves are pinned against
captured fixtures in ``test_host_inspection.py``; what these assert is the TOOL
contract - a non-empty string, never an exception, and the honesty markers
actually reaching the text a model would read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from scufris.host_mcp_server import mcp
from scufris.mcp_host_tools import (
    _format_processes,
    disk_usage,
    host_failed_units,
    host_flake_status,
    host_generation_diff,
    host_journal,
    host_network,
    host_reclaimable_space,
    host_stats,
    host_storage,
    host_thermal,
    host_unit_status,
    host_units,
    host_what_provides,
    list_processes,
)
from scufris_host import ProcessGroup, ProcessList


def test_host_stats_returns_snapshot() -> None:
    stats = host_stats()
    assert isinstance(stats, dict)
    assert stats["hostname"]
    assert "cpu_percent" in stats
    assert "mem" in stats


async def test_host_tool_descriptions_steer_away_from_shell() -> None:
    # The tool descriptions are one of the model's signals; they should explicitly
    # tell it to prefer these over raw shell (the real steering is the prompt
    # preamble in agent.py, but strong descriptions reinforce it).
    desc = {tool.name: (tool.description or "") for tool in await mcp.list_tools()}
    assert "PREFERRED" in desc["host_stats"] or "instead of shell" in desc["host_stats"]
    assert "uname" in desc["host_stats"] and "/proc" in desc["host_stats"]
    assert "PREFER" in desc["disk_usage"]
    assert "PREFER" in desc["list_processes"]
    # Every deep-inspection tool carries the same steering, and names the shell
    # command it replaces so the model can match its instinct to a tool.
    replaces = {
        "host_units": "systemctl list-units",
        "host_failed_units": "systemctl --failed",
        "host_unit_status": "systemctl status",
        "host_journal": "journalctl",
        "host_storage": "df",
        "host_largest_directories": "du",
        "host_reclaimable_space": "nix-collect-garbage",
        "host_network": "iptables",
        "host_thermal": "sensors",
        "host_what_provides": "which",
        "host_generation_diff": "nix store diff-closures",
        "host_flake_status": "flake.lock",
    }
    for name, shell in replaces.items():
        assert "PREFER" in desc[name], f"{name} does not steer away from shell"
        assert shell in desc[name], f"{name} does not name the `{shell}` it replaces"
    # The two tools whose cost is the trap say so, so the model does not poll
    # them. Whitespace-normalised: these phrases sit across a docstring wrap.
    flat = {name: " ".join(text.split()) for name, text in desc.items()}
    assert "tens of seconds" in flat["host_largest_directories"]
    assert "take a minute" in flat["host_reclaimable_space"]
    # And the read-only guarantee is stated where it would be tempting to break.
    assert "read-only" in desc["host_reclaimable_space"].lower()


def test_format_processes_renders_top_groups() -> None:
    plist = ProcessList(
        groups=[
            ProcessGroup(
                name="firefox",
                count=3,
                cpu_percent=42.5,
                mem_rss=3 * 1024 * 1024 * 1024,
                instances=[],
            ),
            ProcessGroup(
                name="python",
                count=1,
                cpu_percent=5.0,
                mem_rss=200 * 1024 * 1024,
                instances=[],
            ),
        ],
        total=57,
    )
    out = _format_processes(plist, limit=1)
    assert "APPLICATION" in out
    assert "total processes: 57" in out
    assert "firefox" in out
    assert "42.5" in out
    assert "3.0GB" in out
    assert "python" not in out  # limited to the top 1 group


def test_disk_usage_returns_table() -> None:
    out = disk_usage()
    # df -h prints a header row and at least the root filesystem.
    assert "Filesystem" in out
    assert "/" in out


def test_list_processes_returns_table() -> None:
    out = list_processes(limit=5)
    assert "APPLICATION" in out
    assert "total processes:" in out


# --- host inspection tools (task 20260729-125024) ----------------------------
#
# These run against the REAL host, like `host_stats` above: they are read-only,
# and a fake would prove only that the fake works. The parsers themselves are
# pinned against captured fixtures in `test_host_inspection.py`; what these
# assert is the TOOL contract - a non-empty string, never an exception, and the
# honesty markers actually reaching the text a model would read.


@pytest.mark.parametrize(
    "call",
    [
        lambda: host_units(state="failed"),
        lambda: host_failed_units(),
        lambda: host_failed_units(scope="user"),
        lambda: host_unit_status("sshd.service"),
        lambda: host_journal(lines=3, since="10 min ago"),
        lambda: host_storage(),
        lambda: host_network(),
        lambda: host_thermal(),
        lambda: host_what_provides("sh"),
        lambda: host_flake_status(),
    ],
    ids=[
        "units",
        "failed-system",
        "failed-user",
        "unit-status",
        "journal",
        "storage",
        "network",
        "thermal",
        "what-provides",
        "flake-status",
    ],
)
def test_host_tools_return_text_and_never_raise(call: Any) -> None:
    out = call()
    assert isinstance(out, str)
    assert out.strip(), "a host tool returned nothing at all"
    # Whatever the outcome, the first line names the report - so a model always
    # knows what it asked about, even when the answer is "unavailable".
    assert len(out.splitlines()) >= 2


def test_host_tools_reject_an_unknown_scope() -> None:
    """A wrong scope is refused, not defaulted: a user unit and a system unit can
    share a name, so silently picking one would answer a different question."""
    for out in (
        host_units(scope="nonsense"),
        host_failed_units(scope="nonsense"),
        host_unit_status("sshd.service", scope="nonsense"),
        host_journal(scope="nonsense"),
    ):
        assert out.startswith("error:")
        assert "system" in out and "user" in out


def test_host_journal_rejects_an_unknown_priority() -> None:
    out = host_journal(priority="urgent", lines=1)
    assert "unknown priority" in out


def test_host_network_states_the_privilege_limit_in_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The declared-vs-live firewall caveat must reach the model, not just the
    model object - this is the text an agent would repeat to the operator.

    Driven through a fixture system tree rather than the live host: an
    `if "unavailable" not in out` guard would make the assertion silently vanish
    on any host whose firewall report degrades, which is precisely the shape a
    test guarding an honesty property must not have.
    """
    from scufris_host import HostInspector

    script = tmp_path / "store" / "abc-firewall-start" / "bin" / "firewall-start"
    script.parent.mkdir(parents=True)
    script.write_text("ip46tables -A nixos-fw -p tcp --dport 22 -j nixos-fw-accept\n")
    unit = tmp_path / "etc" / "systemd" / "system"
    unit.mkdir(parents=True)
    (unit / "firewall.service").write_text(f"ExecStart=@{script} firewall-start\n")
    monkeypatch.setattr(
        "scufris.mcp_host_tools.inspection._inspector",
        lambda: HostInspector(system=tmp_path),
    )

    out = host_network()
    assert "DECLARED" in out
    assert "needs root" in out
    assert "tcp open: 22" in out


def test_host_generation_diff_defaults_to_the_last_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no arguments it compares PREVIOUS -> CURRENT.

    Asserted at the argv, because that is the only place the claim is visible:
    every render path emits a "closure diff ..." title, so a text assertion here
    would be a tautology that passes whichever generations were compared.
    """
    from scufris_host import CommandResult, HostInspector, Outcome

    generations = json.dumps(
        [
            {"generation": 191, "date": "d", "kernelVersion": "k", "current": True},
            {"generation": 190, "date": "d", "kernelVersion": "k", "current": False},
            {"generation": 12, "date": "d", "kernelVersion": "k", "current": False},
        ]
    )
    seen: list[list[str]] = []

    def spy(argv: list[str], *, timeout: float = 10.0) -> CommandResult:
        seen.append(argv)
        stdout = generations if argv[0] == "nixos-rebuild" else "linux: 1 -> 2"
        return CommandResult(argv=argv, outcome=Outcome.OK, stdout=stdout, returncode=0)

    monkeypatch.setattr(
        "scufris.mcp_host_tools.inspection._inspector", lambda: HostInspector(spy)
    )
    out = host_generation_diff()

    diff_argv = [a for a in seen if a[0] == "nix"]
    assert diff_argv, "no closure diff ran"
    joined = " ".join(diff_argv[0])
    # Previous -> current, in that order. Not 12 (the oldest) and not reversed.
    assert "system-190-link" in joined
    assert "system-191-link" in joined
    assert joined.index("system-190-link") < joined.index("system-191-link")
    assert "system-12-link" not in joined
    assert "linux" in out


def test_host_tools_refuse_an_argument_that_would_become_an_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A unit name/pattern starting with '-' is refused, never passed through.

    `shell=False` stops shell injection but NOT option injection: measured on
    this host, `systemctl ... -Hsomeone@elsewhere` makes systemctl open an
    outbound SSH connection to a caller-chosen host. These arguments can come
    from a model that just read attacker-influenced text, so the refusal is
    asserted at the tool boundary AND the argv is checked to prove nothing ran.
    """
    from scufris_host import CommandResult, HostInspector, Outcome

    seen: list[list[str]] = []

    def spy(argv: list[str], *, timeout: float = 10.0) -> CommandResult:
        seen.append(argv)
        return CommandResult(argv=argv, outcome=Outcome.OK, stdout="[]", returncode=0)

    monkeypatch.setattr(
        "scufris.mcp_host_tools.inspection._inspector", lambda: HostInspector(spy)
    )
    hostile = "-Hattacker@evil.example.com"
    for out in (
        host_units(pattern=hostile),
        host_unit_status(hostile),
    ):
        assert "unavailable" in out
        assert "'-'" in out
    assert not seen, f"a command ran with a hostile argument: {seen}"


def test_host_reclaimable_space_never_collects_for_real(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The read-only guarantee at the tool boundary: only an enumerating argv.

    The real command walks the whole store, so the inspector is swapped here -
    what is under test is the argv, not nix.
    """
    from scufris_host import CommandResult, HostInspector, Outcome

    seen: list[list[str]] = []

    def spy(argv: list[str], *, timeout: float = 10.0) -> CommandResult:
        seen.append(argv)
        return CommandResult(
            argv=argv,
            outcome=Outcome.OK,
            stdout="12 store paths would be deleted",
            returncode=0,
        )

    # Swap the INSPECTOR the tool builds, not the module-level `run_command`:
    # HostInspector binds its default runner at definition time, so patching the
    # function afterwards would leave the real one in place and the spy empty.
    monkeypatch.setattr(
        "scufris.mcp_host_tools.inspection._inspector", lambda: HostInspector(spy)
    )
    out = host_reclaimable_space()
    assert seen, "no command ran"
    for argv in seen:
        # --print-dead only ENUMERATES. There is no --delete-older-than here
        # even in dry-run form: that flag also trims profile generations, which
        # would make read-only-ness a property of nix's behaviour rather than
        # of this code.
        assert argv[:3] == ["nix-store", "--gc", "--print-dead"], argv
        assert "--delete-older-than" not in " ".join(argv)
        assert "-d" not in argv
    assert "12 store paths" in out
    assert "not a size" in out


def test_the_agent_has_no_tool_that_approves_a_host_action() -> None:
    """An agent may propose a privileged change. It may never approve one.

    Enforced twice, and this is the cheap half: no tool exists. The other half
    is the middleware refusing the machine bearer token these subprocesses hold
    (tests/test_host_action_api.py). Both, because a tool added for convenience
    would silently undo the expensive one.
    """
    import scufris.mcp_host_tools as server

    names = {
        name
        for name in dir(server)
        if not name.startswith("_") and callable(getattr(server, name))
    }
    approving = {
        name
        for name in names
        if "host" in name and ("approve" in name or "apply" in name)
    }
    assert not approving, f"an agent-facing approval tool exists: {approving}"
    assert "propose_host_action" in names


def _proposal_payload() -> dict[str, Any]:
    """A HostActionRecord as the API returns one, built from the real models."""
    from scufris.host_actions import HostActionRecord
    from scufris_hostd import (
        ActionKind,
        Fingerprint,
        Preview,
        PreviewKind,
        ProposalView,
        Reversal,
        RiskClass,
        Step,
    )

    view = ProposalView(
        id="a" * 32,
        kind=ActionKind.UNIT_RESTART,
        risk=RiskClass.R1,
        args={"unit": "nginx.service"},
        steps=[Step(argv=["systemctl", "restart", "--", "nginx.service"])],
        summary="restart nginx.service",
        preview=Preview(
            kind=PreviewKind.STATE,
            headline="nginx.service is active (running)",
            label="Current state and blast radius - not a prediction.",
            lines=["ActiveState=active", "2 units depend on it"],
        ),
        reversal=Reversal(possible=True, summary="start it again if it stays down"),
        fingerprint=Fingerprint(value="f1", describes="nginx.service"),
        created_at=1.0,
        expires_at=601.0,
    )
    return HostActionRecord(proposal=view).model_dump(mode="json")


@respx.mock
def test_the_host_action_tool_returns_the_rendered_preview_not_json() -> None:
    """The tool hands the model prose, not a blob to paraphrase.

    Its own instruction is "show the operator the preview verbatim", which is
    only possible if the preview text is what comes back (review round 1, R1.11;
    unpinned until review round 2, R2.4).
    """
    from scufris.mcp_host_tools import propose_host_action

    respx.post("http://127.0.0.1:8000/api/host/actions").mock(
        return_value=httpx.Response(200, json=_proposal_payload())
    )

    out = propose_host_action("unit_restart", unit="nginx")

    assert not out.lstrip().startswith("{"), f"the tool returned raw JSON: {out[:80]}"
    assert "Current state and blast radius" in out  # the honesty label, verbatim
    assert "you cannot approve" in out
    assert "the operator must" in out


@respx.mock
def test_a_host_action_tool_error_passes_through_unrendered() -> None:
    """An `error: ...` line is a diagnosable answer; do not turn it into a parse
    failure by insisting on JSON."""
    from scufris.mcp_host_tools import propose_host_action

    respx.post("http://127.0.0.1:8000/api/host/actions").mock(
        return_value=httpx.Response(503, text="no privileged helper configured")
    )

    out = propose_host_action("unit_restart", unit="nginx")
    assert out.startswith("error:")
    assert "no privileged helper configured" in out


@respx.mock
def test_the_host_action_tool_names_the_agent_it_is_running_as(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit's "which agent" field comes from this process's own id.

    The API derives the ACTOR from the credential and will not let a body field
    promote a machine caller (review round 1, R1.6); this half only makes the
    record say something more useful than "an agent".
    """
    from scufris.mcp_host_tools import propose_host_action

    monkeypatch.setenv("SCUFRIS_AGENT_ID", "builder")
    route = respx.post("http://127.0.0.1:8000/api/host/actions").mock(
        return_value=httpx.Response(200, json=_proposal_payload())
    )

    propose_host_action("unit_restart", unit="nginx")

    body = json.loads(route.calls[0].request.content)
    assert body["agent"] == "builder"
    assert body["kind"] == "unit_restart"
    assert body["args"] == {"unit": "nginx"}
