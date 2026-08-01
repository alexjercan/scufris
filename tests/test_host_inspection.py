"""The read-only host inspection package: six domains, one set of properties.

The four properties every domain is held to - typed tools cover it, each tool
degrades explicitly, output is bounded, and empty is distinguishable from broken
- are asserted ACROSS the domains rather than per module, which is why they live
together here. Units, the journal, the network and the render round out the file.

Every parser is driven by output CAPTURED FROM THE REAL HOST
(``tests/fixtures/host/``), per the lesson
``capture-real-cli-output-for-parser-tests``: a parser written against imagined
output is a parser written against the wrong thing. The fixtures were produced
by running the actual commands on the NixOS box this app monitors.

Failures are driven through the same ``Runner`` seam as successes - the whole
reason that seam exists - so "the binary is missing", "permission denied",
"timed out" and "exit 3" are exercised as ordinary inputs rather than as
patched-out internals.

Thermal throttling lives in ``tests/test_host_thermal.py`` and the nix store in
``tests/test_host_nix_store.py``; both import the ``ok``/``broken`` result
helpers from here.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scufris.host import (
    CommandResult,
    FakeRunner,
    HostInspector,
    Outcome,
    Scope,
    closure_diff,
    declared_firewall,
    failed_units,
    largest_directories,
    list_generations,
    list_interfaces,
    list_units,
    read_journal,
    render,
    unit_status,
)
from scufris.host.thermal import thermal_report

FIXTURES = Path(__file__).parent / "fixtures" / "host"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def ok(stdout: str) -> CommandResult:
    return CommandResult(argv=[], outcome=Outcome.OK, stdout=stdout, returncode=0)


def broken(outcome: Outcome, stderr: str = "") -> CommandResult:
    return CommandResult(
        argv=[],
        outcome=outcome,
        stderr=stderr,
        returncode=None if outcome is Outcome.MISSING else 1,
    )


@pytest.fixture
def host_runner() -> FakeRunner:
    """A runner replaying this host's real output for every command."""
    return FakeRunner(
        results={
            "systemctl --system list-units": ok(fixture("systemctl-list-units.json")),
            "systemctl --system show": ok(fixture("systemctl-show-unit.txt")),
            "journalctl": ok(fixture("journalctl.json")),
            "nixos-rebuild list-generations": ok(fixture("nixos-generations.json")),
            "ip -j addr": ok(fixture("ip-addr.json")),
            "nix store diff-closures": ok("linux: 6.18.37 -> 6.18.40\nfoo: 1.0 -> 1.1"),
        }
    )


# --- DoD 1: the six domains are readable through typed tools -----------------


def test_host_inspection_covers_units_logs_and_storage(host_runner: FakeRunner) -> None:
    """Units, logs, storage, network, sensors and generations all parse.

    The Definition-of-Done test for the task. It asserts on the PARSED VALUES
    from the real captured output, not merely that a call returned something -
    a test that only checks `.ok` would pass against a parser that produced
    nothing at all.
    """
    # Units: the real listing has 174 service units on this host.
    units = list_units(host_runner, scope=Scope.SYSTEM, limit=400)
    assert units.ok
    assert units.units, "the captured listing is not empty, so the parse must not be"
    names = {u.name for u in units.units}
    assert "sshd.service" in names
    sshd = next(u for u in units.units if u.name == "sshd.service")
    assert (sshd.load, sshd.active, sshd.sub) == ("loaded", "active", "running")
    assert sshd.description == "SSH Daemon"

    # One unit's status, from `systemctl show` key=value.
    status = unit_status(host_runner, "sshd.service")
    assert status.ok and status.loaded
    assert status.active_state == "active"
    assert status.sub_state == "running"
    assert status.unit_file_state == "enabled"
    assert status.result == "success"
    assert status.main_pid == 1868485
    assert status.restarts == 0
    assert status.memory_bytes == 7045120

    # Logs: journalctl -o json is one object PER LINE, not an array.
    journal = read_journal(host_runner, unit="sshd.service")
    assert journal.ok
    assert len(journal.entries) == 5
    first = journal.entries[0]
    assert first.unit == "sshd.service"
    assert first.timestamp is not None
    assert first.message, "a parsed entry must carry its message"

    # Storage: generations from nixos-rebuild --json.
    generations = list_generations(host_runner)
    assert generations.ok
    assert len(generations.generations) >= 3
    current = generations.current
    assert current is not None and current.number == 191
    assert current.kernel_version == "6.18.40"
    # "Unknown" is nixos-rebuild's placeholder and must not be shown as a rev.
    assert current.configuration_revision == ""
    # Newest first, so the list reads like the rollback menu it describes.
    assert [g.number for g in generations.generations] == sorted(
        (g.number for g in generations.generations), reverse=True
    )

    # Network: interfaces from `ip -j addr`.
    interfaces = list_interfaces(host_runner)
    assert interfaces.ok
    by_name = {i.name: i for i in interfaces.interfaces}
    assert "lo" in by_name
    assert any(a.address == "127.0.0.1" for a in by_name["lo"].addresses)

    # Sensors: driven against a FIXTURE sysfs tree, not the live host. Asserting
    # "temperatures or not ok" against the real machine passes with zero data on
    # a sensorless box, inside the test that is this task's sensor proof.
    with tempfile.TemporaryDirectory() as raw:
        cpu_sysfs = Path(raw)
        for cpu in range(4):
            directory = cpu_sysfs / f"cpu{cpu}" / "thermal_throttle"
            directory.mkdir(parents=True)
            (directory / "core_throttle_count").write_text("2\n")
            (directory / "core_throttle_total_time_ms").write_text("5\n")
            (directory / "package_throttle_count").write_text("78\n")
            (directory / "package_throttle_total_time_ms").write_text("153\n")
        thermal = thermal_report(cpu_sysfs)
    assert thermal.throttling.ok
    assert thermal.throttling.cpus_read == 4
    assert thermal.throttling.core_events == 8
    assert thermal.throttling.package_events == 78
    assert thermal.throttling.throttled
    # Battery and fans are ANSWERED, never silently blank. Two legitimate
    # answers exist and both are fine: "no battery on this host" (a caveat, what
    # this desktop reports) and "battery state is unreadable" (a reason, what
    # the nix build sandbox reports - it has no /sys/class/power_supply at all).
    # What must never happen is a bare empty report with neither, so assert on
    # the MESSAGE rather than on which of the two answers this environment gives.
    for part in (thermal.battery, thermal.fans):
        assert part.available.reason or part.available.caveat, (
            f"{type(part).__name__} said nothing at all about its data"
        )
        assert not part.present

    # Generations: a closure diff with real changes.
    diff = closure_diff(host_runner, 190, 191)
    assert diff.ok and not diff.identical
    assert [c.package for c in diff.changes] == ["linux", "foo"]
    assert diff.changes[0].detail == "6.18.37 -> 6.18.40"


def test_host_inspection_renders_every_domain_without_blank_sections(
    host_runner: FakeRunner,
) -> None:
    """Each renderer emits a titled body, never an empty string."""
    rendered = [
        render.render_units(list_units(host_runner)),
        render.render_unit_status(unit_status(host_runner, "sshd.service")),
        render.render_journal(read_journal(host_runner, unit="sshd.service")),
        render.render_storage(HostInspector(host_runner).storage()),
        render.render_network(HostInspector(host_runner).network()),
        render.render_thermal(thermal_report()),
        render.render_closure_diff(closure_diff(host_runner, 190, 191)),
    ]
    for text in rendered:
        assert text.strip(), "a renderer produced nothing at all"
        assert len(text.splitlines()) >= 2, f"section has a title but no body:\n{text}"


# --- DoD 2: every tool degrades explicitly ----------------------------------


@pytest.mark.parametrize(
    "outcome,stderr,expected",
    [
        (Outcome.MISSING, "", "not installed"),
        (Outcome.DENIED, "Permission denied", "needs privilege"),
        (Outcome.TIMEOUT, "", "did not finish"),
        (Outcome.FAILED, "unit not found", "failed (exit 1)"),
    ],
)
def test_host_inspection_tools_degrade_explicitly(
    outcome: Outcome, stderr: str, expected: str
) -> None:
    """Every command-backed inspection reports WHY, for every failure mode.

    The Definition-of-Done test for honest degradation. It asserts three things
    that must hold together: the report is marked unavailable, it carries a
    non-empty reason naming the failure, and the RENDERED text says "unavailable"
    rather than looking like an empty-but-fine result. The last one is the part
    that actually reaches the model.
    """
    runner = FakeRunner(default=broken(outcome, stderr))
    inspector = HostInspector(
        runner, config_repo=Path("/nonexistent"), system=Path("/nonexistent")
    )

    reports = {
        "units": (inspector.list_units(), render.render_units),
        "failed units": (inspector.failed_units(), render.render_units),
        "unit status": (
            inspector.unit_status("sshd.service"),
            render.render_unit_status,
        ),
        "journal": (inspector.journal(), render.render_journal),
        "generations": (list_generations(runner), None),
        "closure diff": (inspector.closure_diff(1, 2), render.render_closure_diff),
        "reclaimable": (
            inspector.reclaimable_space(),
            render.render_reclaimable,
        ),
        "interfaces": (list_interfaces(runner), None),
    }
    for label, (report, renderer) in reports.items():
        assert not report.ok, f"{label} claimed success on a {outcome} command"
        assert report.available.reason, f"{label} is unavailable with no reason given"
        assert expected in report.available.reason, (
            f"{label} does not name the failure mode: {report.available.reason!r}"
        )
        if renderer is not None:
            text = renderer(report)  # type: ignore[operator]
            assert "unavailable:" in text, (
                f"{label} renders without saying it is unavailable:\n{text}"
            )


def test_unavailable_inspections_never_raise() -> None:
    """Not one inspection raises when everything it needs is missing.

    A tool that raises at the MCP boundary gives the model a traceback instead of
    an answer, so this drives every entry point with a runner that fails and an
    environment with no NixOS paths at all.
    """
    inspector = HostInspector(
        FakeRunner(default=broken(Outcome.MISSING)),
        config_repo=Path("/nonexistent/repo"),
        system=Path("/nonexistent/system"),
        cpu_sysfs=Path("/nonexistent/cpu"),
    )
    calls = [
        lambda: inspector.list_units(),
        lambda: inspector.failed_units(scope=Scope.USER),
        lambda: inspector.unit_status("nope.service"),
        lambda: inspector.journal(unit="nope.service"),
        lambda: inspector.storage(),
        lambda: inspector.generations(),
        lambda: inspector.largest_directories("/nonexistent"),
        lambda: inspector.reclaimable_space(),
        lambda: inspector.network(),
        lambda: inspector.firewall(),
        lambda: inspector.thermal(),
        lambda: inspector.what_provides("definitely-not-a-real-binary"),
        lambda: inspector.profile(),
        lambda: inspector.closure_diff(1, 2),
        lambda: inspector.flake_status(),
        lambda: inspector.overview(),
    ]
    for call in calls:
        call()  # must not raise


def test_missing_firewall_script_says_so_and_names_the_privilege_limit() -> None:
    """The firewall report never implies it read the live table."""
    report = declared_firewall(Path("/nonexistent/system"))
    assert not report.ok
    assert "firewall-start" in report.available.reason
    assert "root" in report.available.reason
    assert report.declared_only is True


def test_unknown_journal_priority_is_refused_not_silently_ignored() -> None:
    """A bad filter must not return unfiltered logs as if the filter applied."""
    runner = FakeRunner(default=ok(""))
    report = read_journal(runner, priority="urgent")
    assert not report.ok
    assert "unknown priority" in report.available.reason
    assert not runner.calls, "journalctl ran despite an invalid priority"


# --- DoD 3: output is bounded -----------------------------------------------


def test_host_inspection_output_is_bounded() -> None:
    """Journal, unit and directory reads cannot exceed their configured caps.

    The Definition-of-Done test for bounding. Each case asks for far MORE than
    the cap and asserts three things: the cap was applied, the report says it was
    truncated, and the rendered text carries the marker - so a model reading the
    result cannot mistake a page for the whole set.
    """
    from scufris.host.journal import MAX_JOURNAL_LINES
    from scufris.host.units import MAX_UNIT_LIMIT

    # Journal: ask for 100000 lines, get the cap - and journalctl is INVOKED
    # with the cap, so the data never even crosses the process boundary.
    entries = "\n".join(
        json.dumps(
            {
                "__REALTIME_TIMESTAMP": str(1785331922137117 + i),
                "MESSAGE": f"line {i}",
                "PRIORITY": "6",
                "_SYSTEMD_UNIT": "test.service",
            }
        )
        for i in range(MAX_JOURNAL_LINES + 50)
    )
    runner = FakeRunner(results={"journalctl": ok(entries)})
    journal = read_journal(runner, lines=100_000)
    assert journal.limit == MAX_JOURNAL_LINES
    assert len(journal.entries) == MAX_JOURNAL_LINES
    assert journal.truncated
    assert "truncated" in render.render_journal(journal)
    invoked = runner.calls[0]
    assert invoked[invoked.index("-n") + 1] == str(MAX_JOURNAL_LINES + 1)

    # Units: the same, over the JSON listing.
    rows = json.dumps(
        [
            {
                "unit": f"unit{i}.service",
                "load": "loaded",
                "active": "active",
                "sub": "running",
                "description": "x",
            }
            for i in range(MAX_UNIT_LIMIT + 25)
        ]
    )
    units = list_units(FakeRunner(results={"systemctl": ok(rows)}), limit=99_999)
    assert units.limit == MAX_UNIT_LIMIT
    assert len(units.units) == MAX_UNIT_LIMIT
    assert units.truncated
    assert "truncated" in render.render_units(units)

    # Directories: du output capped, and the marker rendered.
    du_lines = "\n".join(f"{1000 + i}\t/home/alex/dir{i}" for i in range(200))
    directories = largest_directories(
        FakeRunner(results={"du": ok(du_lines)}), "/", depth=99, limit=99_999
    )
    assert directories.depth <= 3, "depth must be capped, since each level multiplies"
    assert len(directories.directories) == directories.limit
    assert directories.truncated
    assert "truncated" in render.render_largest_directories(directories)


def test_journal_byte_cap_stops_a_few_enormous_lines() -> None:
    """The line cap alone does not bound output; the byte cap must also hold.

    Ten lines is well under the 100-line default, so only the byte budget can
    stop this - which is exactly the case a line-count cap misses.
    """
    from scufris.host.journal import MAX_JOURNAL_BYTES

    huge = "x" * 20_000
    entries = "\n".join(
        json.dumps({"MESSAGE": huge, "PRIORITY": "6", "__REALTIME_TIMESTAMP": "1"})
        for _ in range(10)
    )
    report = read_journal(FakeRunner(results={"journalctl": ok(entries)}))
    assert report.truncated and report.bytes_truncated
    assert len(report.entries) < 10
    total = sum(len(e.message) for e in report.entries)
    assert total <= MAX_JOURNAL_BYTES
    assert "byte budget" in render.render_journal(report)


# --- DoD 4: empty is distinguishable from broken ----------------------------


def test_host_inspection_distinguishes_empty_from_broken() -> None:
    """A successful-but-empty result is never rendered as a blank.

    The three cases where an empty rendering would be actively misleading: no
    failed units (good news), an identical closure (nix prints NOTHING and exits
    0 - measured on this host), and a socket whose owner needs privilege.
    """
    # 1. No failed units: the good-news case that must be stated, not implied.
    empty_units = failed_units(FakeRunner(results={"systemctl": ok("[]")}))
    assert empty_units.ok
    assert empty_units.units == []
    text = render.render_units(empty_units)
    assert "no failed system units" in text
    assert "nothing is in a failed state" in text
    assert "unavailable" not in text

    # ... and the broken case renders differently, which is the entire point.
    broken_units = failed_units(FakeRunner(default=broken(Outcome.DENIED, "denied")))
    assert render.render_units(broken_units) != text
    assert "unavailable:" in render.render_units(broken_units)

    # 2. The closure-diff trap: empty stdout, exit 0, identical closures.
    identical = closure_diff(FakeRunner(results={"nix": ok("")}), 190, 191)
    assert identical.ok
    assert identical.identical is True
    assert identical.changes == []
    rendered = render.render_closure_diff(identical)
    assert "no closure change" in rendered
    assert "unavailable" not in rendered

    # ... versus a diff that genuinely failed: same empty output, different exit.
    failed_diff = closure_diff(
        FakeRunner(default=broken(Outcome.FAILED, "path does not exist")), 190, 191
    )
    assert not failed_diff.ok
    assert failed_diff.identical is False
    assert "unavailable:" in render.render_closure_diff(failed_diff)

    # 3. A socket whose owner is not visible says so instead of rendering blank.
    from scufris.host.network import ListeningSocket, SocketReport

    report = SocketReport(
        sockets=[
            ListeningSocket(
                protocol="tcp", address="0.0.0.0", port=22, owner_hidden=True
            )
        ]
    )
    lines = "\n".join(render.render_sockets(report))
    assert "owner not visible without privilege" in lines


def test_a_unit_that_does_not_exist_is_reported_as_such() -> None:
    """`systemctl show` succeeds for an unknown unit; the report must not pretend."""
    runner = FakeRunner(
        results={
            "systemctl": ok(
                "Id=nope.service\nLoadState=not-found\nActiveState=inactive"
            )
        }
    )
    status = unit_status(runner, "nope.service")
    assert status.ok  # systemd answered
    assert not status.loaded
    assert "no unit named" in status.available.caveat
    assert "no such unit is loaded" in render.render_unit_status(status)


def test_declared_firewall_parses_the_real_nixos_script(tmp_path: Path) -> None:
    """Parse a real firewall-start script shape, including per-interface rules.

    Captured from this host: global openings plus an interface-scoped one, and a
    port declared twice (11433), which the summary must not repeat.
    """
    script = tmp_path / "store" / "abc-firewall-start" / "bin" / "firewall-start"
    script.parent.mkdir(parents=True)
    script.write_text(
        "\n".join(
            [
                "ip46tables -A nixos-fw -i lo -j nixos-fw-accept",
                "ip46tables -A nixos-fw -m conntrack --ctstate ESTABLISHED,RELATED "
                "-j nixos-fw-accept",
                "ip46tables -A nixos-fw -p tcp --dport 22 -j nixos-fw-accept ",
                "ip46tables -A nixos-fw -p tcp --dport 11433 -j nixos-fw-accept ",
                "ip46tables -A nixos-fw -i enp4s0 -p tcp --dport 11433 "
                "-j nixos-fw-accept ",
                "ip46tables -A nixos-fw -p udp --dport 27031:27035 -j nixos-fw-accept ",
                "ip46tables -A nixos-fw -j nixos-fw-log-refuse",
            ]
        )
    )
    unit = tmp_path / "etc" / "systemd" / "system"
    unit.mkdir(parents=True)
    (unit / "firewall.service").write_text(f"ExecStart=@{script} firewall-start\n")

    report = declared_firewall(tmp_path)
    assert report.ok
    assert report.declared_only is True
    # Structural rules (loopback, conntrack, final refuse) are not openings.
    assert all(rule.ports for rule in report.rules)
    assert report.allowed_tcp == ["22", "11433"], "a port opened twice is listed once"
    assert report.allowed_udp == ["27031:27035"]
    assert any(rule.interface == "enp4s0" for rule in report.rules)
    text = "\n".join(render.render_firewall(report))
    assert "DECLARED" in text
    assert "needs root" in text


def test_a_partially_readable_du_keeps_its_data_with_a_caveat() -> None:
    """du exits non-zero for one unreadable subdir while printing the rest.

    Discarding a real answer over an unreadable dotfile directory would be worse
    than reporting it with the caveat attached.
    """
    result = CommandResult(
        argv=["du"],
        outcome=Outcome.FAILED,
        stdout="4096\t/home/alex/a\n8192\t/home/alex/b\n",
        stderr="du: cannot read directory '/home/alex/.cache/x': Permission denied",
        returncode=1,
    )
    report = largest_directories(FakeRunner(default=result), "/", limit=10)
    assert report.ok, "a partial answer is still an answer"
    assert [d.path for d in report.directories] == ["/home/alex/b", "/home/alex/a"]
    assert "could not be read" in report.available.caveat
    assert "could not be read" in render.render_largest_directories(report)


def test_overview_holds_both_scopes_and_the_cheap_reports_only(
    host_runner: FakeRunner,
) -> None:
    """The dashboard snapshot covers system AND user units, and nothing slow.

    scufris itself is a USER unit on this host, so a system-only overview would
    miss the app's own failure.
    """
    overview = HostInspector(host_runner).overview()
    assert overview.failed_system_units.scope == Scope.SYSTEM
    assert overview.failed_user_units.scope == Scope.USER
    invoked = " ".join(" ".join(argv) for argv in host_runner.calls)
    # Named by the ARGUMENT the store walk actually uses. Asserting on
    # "nix-collect-garbage" can no longer fail - nothing invokes that command
    # any more - which is the vacuous shape round 1 flagged elsewhere.
    assert "--print-dead" not in invoked, "the store walk must not be polled"
    assert "du" not in invoked, "the directory walk must not be polled"


def test_a_single_oversized_journal_line_is_never_reported_as_an_empty_window() -> None:
    """One message bigger than the whole byte budget must not read as "nothing".

    Round 1, MAJOR: the budget is decremented before the first entry is
    appended, so a 50 KB message emptied the entry list while `truncated` was
    set - and the renderer's empty branch then printed "the window is empty, not
    broken" about data that had been read and thrown away.
    """
    from scufris.host.journal import MAX_JOURNAL_BYTES

    huge = "x" * (MAX_JOURNAL_BYTES + 10_000)
    entry = json.dumps(
        {"MESSAGE": huge, "PRIORITY": "3", "__REALTIME_TIMESTAMP": "1785331922137117"}
    )
    report = read_journal(FakeRunner(results={"journalctl": ok(entry)}))
    assert report.ok
    assert report.truncated and report.bytes_truncated
    # The entry is KEPT and clipped, rather than dropped entirely.
    assert len(report.entries) == 1
    assert "[message cut to fit]" in report.entries[0].message
    text = render.render_journal(report)
    assert "the window is empty" not in text
    assert "cut" in text


def test_an_all_oversized_journal_read_still_says_it_is_not_empty() -> None:
    """Belt and braces on the renderer: even with zero entries and truncated set,
    the text must contradict "empty" rather than assert it."""
    from scufris.host.journal import JournalReport

    report = JournalReport(truncated=True, bytes_truncated=True, total_seen=7, limit=50)
    text = render.render_journal(report)
    assert "NOT EMPTY" in text
    assert "the window is empty" not in text


def test_journal_truncation_survives_an_unparseable_line() -> None:
    """Unparsed lines do not consume the cap, so the entry count alone
    under-reports truncation and presents a full page as the complete set."""
    good = "\n".join(
        json.dumps({"MESSAGE": f"m{i}", "PRIORITY": "6", "__REALTIME_TIMESTAMP": "1"})
        for i in range(5)
    )
    # 6 raw lines for a cap of 5, the FIRST of which is garbage -> exactly 5
    # entries. Order matters: a trailing garbage line is never reached, because
    # the cap is hit first.
    raw = "not json at all\n" + good
    report = read_journal(FakeRunner(results={"journalctl": ok(raw)}), lines=5)
    assert len(report.entries) == 5
    assert report.unparsed == 1
    assert report.truncated, "a full page was presented as the complete set"
    assert "truncated" in render.render_journal(report)


def test_a_unit_argument_that_would_become_an_option_is_refused() -> None:
    """Option injection, not shell injection: `shell=False` does not stop it.

    Measured on this host: `systemctl ... -Hsomeone@elsewhere` opens an outbound
    SSH connection to a caller-chosen host with the service user's credentials.
    The argument can come from a model that just read attacker-influenced text,
    so nothing may run at all.
    """
    runner = FakeRunner(results={"systemctl": ok("[]")})
    hostile = "-Hattacker@evil.example.com"

    units = list_units(runner, pattern=hostile)
    assert not units.ok
    assert "'-'" in units.available.reason
    status = unit_status(runner, hostile)
    assert not status.ok
    assert "'-'" in status.available.reason
    assert not runner.calls, f"a command ran with a hostile argument: {runner.calls}"


def test_positionals_are_passed_after_a_double_dash() -> None:
    """The second guard: even a permitted operand is separated from the options.

    Verified live that `systemctl ... -- <pattern>` treats the pattern as an
    operand where the bare form parsed it as a flag.
    """
    runner = FakeRunner(results={"systemctl": ok("[]"), "du": ok("")})
    list_units(runner, pattern="nginx*")
    unit_status(runner, "sshd.service")
    argvs = [argv for argv in runner.calls if argv[0] == "systemctl"]
    assert len(argvs) == 2
    for argv in argvs:
        assert "--" in argv, argv
        # The operand is the LAST element, after the separator.
        assert argv.index("--") == len(argv) - 2, argv
