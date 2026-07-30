"""The scheduled host checks and the digest: the one thing here that fires by itself.

Two levels, deliberately:

- the SCHEDULER and the RENDERER are driven directly with an injected clock and a
  fake run, because what they are is timing and judgement - no waiting, no host;
- the whole PATH (a schedule fires -> the checks read -> a digest is rendered ->
  Telegram gets it -> a breach becomes a proposal in the real approval queue) is
  driven through the real app with a real `scufris-hostd` and respx standing in for
  the Bot API, because "arrives without any operator interaction" is only true if
  nothing in between needed a person.

The checks themselves read a FakeRunner replaying this box's real command output
(`test_host_actions.host_runner`), so a threshold is judged against the shapes the
real inspector returns rather than against a mock of the judgement.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
import pytest
import respx
from conftest import _Helper
from fastapi.testclient import TestClient
from test_host_action_api import ORIGIN, _login, _settings

from scufris.app import create_app
from scufris.auth import CSRF_HEADER
from scufris.checks import (
    ESCALATABLE,
    CheckResult,
    CheckRun,
    CheckState,
    check_scufris,
    escalation_for,
    run_checks,
)
from scufris.config import Settings
from scufris.digest import DigestStore, render_digest
from scufris.health import AgentHealth, HealthCheck
from scufris.host import HostInspector
from scufris.hostd.actions import ActionKind
from scufris.metrics import Collector
from scufris.scheduler import (
    DAILY,
    WATCH,
    HostScheduler,
    SchedulerStore,
    next_daily_due,
)

API = "https://api.telegram.org/botTEST"
CHAT = 4242

# A Monday at 09:30 local, so "08:00" is always in the past for the same day and the
# arithmetic below is readable.
NOW = time.mktime(time.struct_time((2026, 7, 27, 9, 30, 0, 0, 208, -1)))


def _clock(start: float = NOW) -> tuple[Callable[[], float], Callable[[float], None]]:
    """A clock the test moves by hand, so nothing here sleeps."""
    state = {"t": start}

    def now() -> float:
        return state["t"]

    def advance(seconds: float) -> None:
        state["t"] += seconds

    return now, advance


def _settings_for(tmp_path: Path, **kwargs: Any) -> Settings:
    base: dict[str, Any] = {
        "state_dir": tmp_path / "state",
        "web_dist": tmp_path / "absent",
        "_env_file": None,
    }
    base.update(kwargs)
    return Settings(**base)


def _result(name: str, state: CheckState, headline: str = "x") -> CheckResult:
    return CheckResult(name=name, state=state, headline=headline)


def _run(*results: CheckResult, at: float = NOW) -> CheckRun:
    return CheckRun(at=at, results=list(results))


# --- the scheduler ----------------------------------------------------------


async def test_a_schedule_fires_when_it_is_due_and_not_before(tmp_path: Path) -> None:
    now, advance = _clock()
    ran: list[str] = []

    async def run(name: str) -> str:
        ran.append(name)
        return "ran"

    scheduler = HostScheduler(
        SchedulerStore(tmp_path),
        run=run,
        watch_interval=lambda: 900.0,
        daily_at=lambda: "08:00",
        watch_enabled=lambda: True,
        daily_enabled=lambda: True,
        muted_until=lambda: 0.0,
        clock=now,
    )

    # First sight ARMS both schedules and runs nothing: a fresh boot (or a restart
    # loop) must not fire a pass at startup.
    assert await scheduler.tick() == []
    assert ran == []
    states = {state.name: state for state in scheduler.states()}
    assert states[WATCH].next_due == pytest.approx(NOW + 900)
    assert states[DAILY].next_due == pytest.approx(next_daily_due("08:00", now=NOW))
    assert "nothing has run yet" in states[WATCH].last_result

    # Not yet due: still nothing.
    advance(899)
    assert await scheduler.tick() == []

    # Due: it runs once, and re-arms.
    advance(2)
    assert await scheduler.tick() == [WATCH]
    assert ran == [WATCH]
    watch = {s.name: s for s in scheduler.states()}[WATCH]
    assert watch.runs == 1
    assert watch.last_run == pytest.approx(NOW + 901)
    assert watch.next_due == pytest.approx(NOW + 901 + 900)


async def test_schedules_survive_restart_without_stampede(tmp_path: Path) -> None:
    """A window missed while the app was down is recorded, not replayed.

    The failure this pins: an app down for six hours coming back and delivering
    twenty-four `watch` digests at once, which is how a useful feature becomes a
    muted one.
    """
    now, advance = _clock()
    ran: list[str] = []

    async def run(name: str) -> str:
        ran.append(name)
        return "ran"

    def build(clock: Callable[[], float]) -> HostScheduler:
        return HostScheduler(
            SchedulerStore(tmp_path),
            run=run,
            watch_interval=lambda: 900.0,
            daily_at=lambda: "08:00",
            watch_enabled=lambda: True,
            daily_enabled=lambda: True,
            muted_until=lambda: 0.0,
            clock=clock,
        )

    first = build(now)
    await first.tick()  # arms
    advance(901)
    await first.tick()  # one real run
    assert ran == [WATCH]
    armed = {s.name: s.next_due for s in first.states()}

    # The app goes away for six hours and comes back. A SECOND scheduler over the
    # same state dir is what a restart is.
    advance(6 * 3600)
    second = build(now)
    restored = {s.name: s for s in second.states()}
    assert restored[WATCH].next_due == pytest.approx(armed[WATCH])
    assert restored[WATCH].runs == 1  # the history survived

    ran.clear()
    assert await second.tick() == []  # the window is long past: skipped, not fired
    assert ran == []
    watch = {s.name: s for s in second.states()}[WATCH]
    assert watch.missed == 1
    assert "window missed" in watch.last_result
    assert watch.next_due == pytest.approx(now() + 900)

    # And the next real window runs exactly once - no backlog.
    advance(901)
    assert await second.tick() == [WATCH]
    assert ran == [WATCH]


async def test_a_run_never_overlaps_itself(tmp_path: Path) -> None:
    now, advance = _clock()
    release = asyncio.Event()
    started = asyncio.Event()
    calls: list[str] = []

    async def run(name: str) -> str:
        calls.append(name)
        started.set()
        await release.wait()
        return "ran"

    scheduler = HostScheduler(
        SchedulerStore(tmp_path),
        run=run,
        watch_interval=lambda: 900.0,
        daily_at=lambda: "08:00",
        watch_enabled=lambda: True,
        daily_enabled=lambda: False,
        muted_until=lambda: 0.0,
        clock=now,
    )
    await scheduler.tick()  # arms
    advance(901)
    first = asyncio.create_task(scheduler.tick())
    await asyncio.wait_for(started.wait(), timeout=5)

    # A second tick while the first run is in flight records a skip and does NOT
    # start a second pass over the same subprocess reads.
    advance(901)
    assert await scheduler.tick() == []
    assert calls == [WATCH]
    skipped = {s.name: s for s in scheduler.states()}[WATCH]
    assert "previous run was still going" in skipped.last_result
    assert skipped.missed == 1

    release.set()
    await asyncio.wait_for(first, timeout=5)
    assert calls == [WATCH]


async def test_a_disabled_schedule_does_nothing_and_a_failing_run_is_recorded(
    tmp_path: Path,
) -> None:
    now, advance = _clock()
    enabled = {"watch": False}

    async def run(name: str) -> str:
        raise RuntimeError("the checks exploded")

    scheduler = HostScheduler(
        SchedulerStore(tmp_path),
        run=run,
        watch_interval=lambda: 900.0,
        daily_at=lambda: "08:00",
        watch_enabled=lambda: enabled["watch"],
        daily_enabled=lambda: False,
        muted_until=lambda: 0.0,
        clock=now,
    )
    advance(10_000)
    assert await scheduler.tick() == []  # disabled: not even armed into running

    enabled["watch"] = True
    await scheduler.tick()  # arms
    advance(901)
    assert await scheduler.tick() == [WATCH]
    # A run that raises is RECORDED, not propagated: a scheduler that dies on one bad
    # pass is silent in exactly the way this feature exists to prevent.
    state = {s.name: s for s in scheduler.states()}[WATCH]
    assert "the run failed" in state.last_result
    assert "the checks exploded" in state.last_result
    assert state.next_due > now()


def test_the_daily_time_is_local_and_a_typo_does_not_disable_it() -> None:
    due = next_daily_due("08:00", now=NOW)
    assert due > NOW  # 09:30 is past 08:00, so it is tomorrow
    assert time.localtime(due).tm_hour == 8
    assert time.localtime(due).tm_min == 0
    later = next_daily_due("23:45", now=NOW)
    assert time.localtime(later).tm_hour == 23
    # A malformed value costs the hour, not the heartbeat.
    assert time.localtime(next_daily_due("nonsense", now=NOW)).tm_hour == 8


# --- the digest -------------------------------------------------------------


def test_the_boring_case_is_silent_except_for_the_daily_line() -> None:
    """`watch` says nothing when there is nothing; `daily` always says one line."""
    quiet = _run(
        _result("disk", CheckState.OK, "disks are fine"),
        _result("units", CheckState.OK, "nothing is in a failed state"),
    )
    assert render_digest(quiet, schedule=WATCH) is None

    daily = render_digest(quiet, schedule=DAILY, always=True)
    assert daily is not None
    assert daily.verdict == "ok"
    assert len(daily.text.splitlines()) == 1
    assert "all clear" in daily.text
    assert "2 check(s)" in daily.text


def test_an_unreadable_check_never_hides_inside_an_all_clear() -> None:
    run = _run(
        _result("disk", CheckState.OK),
        _result("thermal", CheckState.UNAVAILABLE, "no sensors"),
    )
    # UNAVAILABLE wants attention, so `watch` speaks...
    watch = render_digest(run, schedule=WATCH)
    assert watch is not None
    assert "no sensors" in watch.text
    # ...and when a check IS unreadable, the daily all-clear names it rather than
    # counting it as a pass.
    daily = render_digest(run, schedule=DAILY, always=True)
    assert daily is not None
    assert "thermal" in daily.text

    clean = render_digest(_run(_result("disk", CheckState.OK)), schedule=DAILY, always=True)
    assert clean is not None
    assert "could not be read" not in clean.text


def test_the_digest_leads_with_the_worst_and_says_what_changed() -> None:
    before = {"disk": "ok", "units": "crit", "thermal": "ok"}
    run = _run(
        _result("disk", CheckState.WARN, "/ is 88% full"),
        _result("units", CheckState.OK, "nothing is in a failed state"),
        _result("thermal", CheckState.CRIT, "the CPU is at 97C"),
    )
    digest = render_digest(run, previous=before, schedule=WATCH)
    assert digest is not None
    lines = digest.text.splitlines()
    # The crit leads, the warn follows.
    assert "97C" in lines[0]
    assert any("88% full" in line for line in lines[1:])
    # A recovery is its own line - otherwise the only way to learn something got
    # better is to notice it stopped being mentioned.
    assert any("recovered since the last digest: units" in line for line in lines)
    assert any("new since the last digest" in line and "thermal" in line for line in lines)
    assert digest.verdict == "attention"


def test_a_recovery_alone_is_worth_a_message() -> None:
    before = {"disk": "crit"}
    run = _run(_result("disk", CheckState.OK, "disks are fine"))
    digest = render_digest(run, previous=before, schedule=WATCH)
    assert digest is not None
    assert "recovered" in digest.text
    assert digest.verdict == "ok"


def test_the_digest_store_survives_a_restart_and_stays_bounded(tmp_path: Path) -> None:
    store = DigestStore(tmp_path, max_digests=3)
    for index in range(5):
        digest = render_digest(
            _run(_result("disk", CheckState.WARN, f"warning {index}")),
            schedule=WATCH,
        )
        assert digest is not None
        store.add(digest)
    assert len(store.list()) == 3
    assert "warning 4" in store.list()[0].text  # newest first

    fresh = DigestStore(tmp_path, max_digests=3)
    assert [d.text for d in fresh.list()] == [d.text for d in store.list()]
    assert fresh.last_states() == {"disk": "warn"}

    # A corrupt file costs the history, never the boot.
    (tmp_path / "digests.json").write_text("{ not json")
    assert DigestStore(tmp_path).list() == []


def test_delivery_is_recorded_on_the_digest(tmp_path: Path) -> None:
    store = DigestStore(tmp_path)
    digest = store.add(
        render_digest(_run(_result("disk", CheckState.CRIT, "full")), schedule=WATCH)
        or pytest.fail("expected a digest")
    )
    store.mark_delivered(digest, error="telegram exploded")
    reloaded = DigestStore(tmp_path).latest()
    assert reloaded is not None
    assert reloaded.delivered is False
    assert reloaded.delivery_error == "telegram exploded"


# --- the checks -------------------------------------------------------------


async def test_digest_survives_a_failing_check(tmp_path: Path) -> None:
    """A check that raises and a check that hangs both become named failures.

    The whole digest must not vanish because one read broke - and the failure has to
    be NAMED, or the operator reads a short digest as good news.
    """
    settings = _settings_for(tmp_path)

    def boom(inspector: Any, settings: Any) -> CheckResult:
        raise RuntimeError("sensors are on fire")

    def sleeper(inspector: Any, settings: Any) -> CheckResult:
        time.sleep(2)
        return _result("slow", CheckState.OK)

    import scufris.checks as checks_mod

    original = checks_mod.HOST_CHECKS
    checks_mod.HOST_CHECKS = (  # type: ignore[assignment]
        ("disk", lambda i, s: _result("disk", CheckState.OK, "disks are fine")),
        ("thermal", boom),
        ("slow", sleeper),
    )
    try:
        run = await run_checks(None, settings, timeout=0.2)  # type: ignore[arg-type]
    finally:
        checks_mod.HOST_CHECKS = original  # type: ignore[assignment]

    states = {r.name: r for r in run.results}
    assert states["disk"].state is CheckState.OK
    assert states["thermal"].state is CheckState.FAILED
    assert "sensors are on fire" in states["thermal"].headline
    assert states["slow"].state is CheckState.FAILED
    assert "timed out" in states["slow"].headline

    # And the digest still goes out, naming both.
    digest = render_digest(run, schedule=WATCH)
    assert digest is not None
    assert "sensors are on fire" in digest.text or "thermal" in digest.text
    assert "slow" in digest.text


async def test_the_checks_read_the_real_inspector_shapes(tmp_path: Path) -> None:
    """Every check runs against the command output this box really produces.

    Not a mock of the judgement: `host_runner` replays captured `systemctl`,
    `nixos-rebuild` and `nix-store` output, so a threshold is exercised against the
    shapes the inspector actually returns.
    """
    from test_host_actions import host_runner

    settings = _settings_for(tmp_path, check_disk_warn_percent=1.0)
    inspector = HostInspector(runner=host_runner(), config_repo=tmp_path)
    run = await run_checks(inspector, settings)

    names = {result.name for result in run.results}
    assert names == {"disk", "units", "thermal", "store", "flake"}
    # Nothing raised, and nothing came back FAILED.
    assert not [r for r in run.results if r.state is CheckState.FAILED]
    # The disk threshold was set absurdly low, so it must have fired: a check that
    # cannot fire is not a check.
    disk = {r.name: r for r in run.results}["disk"]
    assert disk.state in (CheckState.WARN, CheckState.CRIT, CheckState.UNAVAILABLE)


def test_the_scufris_check_reads_the_same_health_the_dashboard_shows() -> None:
    healthy = AgentHealth(scufris_version="9.9.9", backend="mock", checks=[])
    assert check_scufris(healthy).state is CheckState.OK
    assert "9.9.9" in check_scufris(healthy).headline

    degraded = AgentHealth(
        scufris_version="9.9.9",
        backend="mock",
        checks=[
            HealthCheck(name="codex auth", status="warn", detail="unknown"),
            HealthCheck(name="state dir", status="error", detail="not writable"),
        ],
    )
    result = check_scufris(degraded)
    assert result.state is CheckState.CRIT
    assert "state dir" in result.headline
    assert result.facts == {"errors": 1, "warnings": 1}


def test_only_the_cleanup_verbs_can_be_escalated() -> None:
    """A threshold may propose a store collection and nothing else.

    The allowlist is the guarantee: a check must never be able to ask for a service
    restart or a configuration switch, because a threshold cannot judge what those
    cost.
    """
    assert ESCALATABLE == {ActionKind.GC_STORE}
    escalation = escalation_for(ActionKind.GC_STORE, {}, because="the store is full")
    assert escalation.kind is ActionKind.GC_STORE
    for kind in ActionKind:
        if kind in ESCALATABLE:
            continue
        with pytest.raises(ValueError, match="not escalatable"):
            escalation_for(kind, {}, because="no")


# --- the whole path ---------------------------------------------------------


class _Api:
    """Recorded Bot API traffic."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.fail = False

    def texts(self) -> list[str]:
        return [str(payload.get("text", "")) for payload in self.sent]

    def install(self, router: respx.Router) -> None:
        def send(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            if self.fail:
                return httpx.Response(500, json={"ok": False})
            self.sent.append(payload)
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

        router.post(f"{API}/sendMessage").mock(side_effect=send)
        router.post(f"{API}/editMessageText").mock(
            return_value=httpx.Response(200, json={"ok": True, "result": {}})
        )
        router.post(f"{API}/answerCallbackQuery").mock(
            return_value=httpx.Response(200, json={"ok": True, "result": True})
        )
        # One 500 and the poll loop backs off, rather than busy-spinning on respx.
        router.post(f"{API}/getUpdates").mock(
            return_value=httpx.Response(500, json={"ok": False})
        )


@pytest.fixture
def api() -> Iterator[_Api]:
    recorder = _Api()
    with respx.mock(assert_all_called=False) as router:
        recorder.install(router)
        yield recorder


def _app_with_checks(
    tmp_path: Path, helper: _Helper, fake_collector: Collector, **kw: Any
) -> Any:
    """The real app, with a FAKE host to read.

    The inspector is injected for the same reason the config builder is: a real check
    pass walks the nix store and shells out to systemctl, so driving it for real
    would make each of these tests tens of seconds long and dependent on the machine
    running them. `host_runner` replays this box's captured output, so the thresholds
    are still judged against real shapes.
    """
    from test_host_actions import host_runner

    settings = _settings(
        tmp_path,
        helper,
        telegram_bot_token="TEST",
        telegram_allowed_chat_ids=[CHAT],
        **kw,
    )
    return create_app(
        collector=fake_collector,
        settings=settings,
        host_inspector=HostInspector(runner=host_runner(), config_repo=tmp_path),
    )


def _drive(client: TestClient, app: Any, schedule: str = WATCH) -> str:
    """Run one schedule on the APP's loop, as its own scheduler would."""
    import functools

    return client.portal.call(  # type: ignore[union-attr]
        functools.partial(app.state.host_scheduler.run_now, schedule)
    )


def test_scheduled_host_digest_is_delivered(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """The epic's Done Means 5: it arrives with no operator interaction.

    The daily schedule always speaks, so this is the one that must deliver whatever
    the machine looks like.
    """
    app = _app_with_checks(tmp_path, helper, fake_collector)
    client = make_client(app)
    result = _drive(client, app, DAILY)

    assert "delivered" in result, result
    assert api.texts(), "nothing was sent to the operator"
    body = api.texts()[-1]
    assert "all clear" in body or "-" in body
    # And it is readable afterwards, without asking Telegram.
    _login(client)
    view = client.get("/api/host/digests").json()
    assert view["enabled"] is True
    assert view["digests"], "the digest was not recorded"
    assert view["digests"][0]["delivered"] is True
    assert view["digests"][0]["schedule"] == DAILY
    daily = {s["name"]: s for s in view["schedules"]}[DAILY]
    assert daily["runs"] == 1
    assert "delivered" in daily["last_result"]


def test_a_delivery_failure_keeps_the_digest(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """Telegram being down costs the message, not the record."""
    app = _app_with_checks(tmp_path, helper, fake_collector)
    client = make_client(app)
    api.fail = True
    result = _drive(client, app, DAILY)

    assert "delivery failed" in result, result
    _login(client)
    view = client.get("/api/host/digests").json()
    assert view["digests"], "the digest was lost with the delivery"
    assert view["digests"][0]["delivered"] is False
    assert view["digests"][0]["delivery_error"]
    assert view["digests"][0]["text"]


def test_a_muted_window_still_runs_and_records(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """A mute is "stop messaging me", not "stop watching"."""
    app = _app_with_checks(
        tmp_path,
        helper,
        fake_collector,
        host_digest_muted_until=time.time() + 3600,
    )
    client = make_client(app)
    result = _drive(client, app, DAILY)

    assert "muted" in result, result
    assert api.texts() == []
    _login(client)
    view = client.get("/api/host/digests").json()
    assert view["digests"][0]["delivery_error"] == "muted"
    assert view["digests"][0]["text"], "the digest was not written"


def test_check_escalation_requires_approval(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """A breach proposes; it never applies.

    Driven with thresholds low enough that the store check must breach, and with
    escalation switched ON - the point being that even then the action waits for a
    decision like any other.
    """
    app = _app_with_checks(
        tmp_path,
        helper,
        fake_collector,
        check_escalate_gc=True,
        check_store_dead_paths=1,
        check_disk_warn_percent=0.0,
    )
    client = make_client(app)
    csrf = _login(client)
    _drive(client, app, WATCH)

    queue = client.get("/api/host/actions").json()
    assert queue, "the breach did not propose anything"
    proposal = queue[0]["proposal"]
    assert proposal["kind"] == "gc_store"
    assert queue[0]["decision"] == "pending"
    # NOTHING ran: the executor is untouched until an operator approves.
    assert helper.executor.calls == []

    # And it is a normal proposal - approving it needs the acknowledgement a one-way
    # action always needs.
    action_id = proposal["id"]
    headers = {"Origin": ORIGIN, CSRF_HEADER: csrf}
    assert (
        client.post(
            f"/api/host/actions/{action_id}/approve", headers=headers
        ).status_code
        == 422
    )
    assert helper.executor.calls == []


def test_escalation_is_off_by_default(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """The same breach with the default settings proposes nothing at all."""
    app = _app_with_checks(
        tmp_path,
        helper,
        fake_collector,
        check_store_dead_paths=1,
        check_disk_warn_percent=0.0,
    )
    client = make_client(app)
    _login(client)
    _drive(client, app, WATCH)

    assert client.get("/api/host/actions").json() == []
    assert helper.executor.calls == []


def test_run_now_returns_immediately_and_is_operator_only(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """The button starts the run and returns; a machine token cannot press it.

    Returning immediately is not a nicety: a full pass walks the nix store, and an
    HTTP request holding that open is how a route sweep turned into 38 seconds of
    real host I/O while this was being built.
    """
    app = _app_with_checks(tmp_path, helper, fake_collector)
    client = make_client(app)
    machine = {
        "Authorization": f"Bearer {app.state.api_token}",
        "Origin": ORIGIN,
    }
    refused = client.post("/api/host/digests/run", headers=machine)
    assert refused.status_code == 403
    assert "operator session" in refused.json()["detail"]

    csrf = _login(client)
    headers = {"Origin": ORIGIN, CSRF_HEADER: csrf}
    started = time.monotonic()
    accepted = client.post("/api/host/digests/run?schedule=daily", headers=headers)
    assert accepted.status_code == 202
    assert time.monotonic() - started < 2.0, "the request waited for the run"

    for _ in range(300):
        view = client.get("/api/host/digests").json()
        if view["digests"]:
            break
        time.sleep(0.02)
    assert view["digests"], "the started run never produced a digest"

    bad = client.post("/api/host/digests/run?schedule=nope", headers=headers)
    assert bad.status_code == 422


def test_the_scheduler_is_started_by_the_app(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    api: _Api,
) -> None:
    """The loop runs for the app's lifetime - and arms the schedules without firing.

    A fresh boot must not perform a pass: that was a build-time correction, and this
    is the guard for it.
    """
    app = _app_with_checks(tmp_path, helper, fake_collector)
    with TestClient(app):
        task = app.state.host_checks_task
        assert task is not None and not task.done()
        for _ in range(200):
            states = {s.name: s for s in app.state.host_scheduler.states()}
            if states[WATCH].next_due:
                break
            time.sleep(0.01)
        assert states[WATCH].next_due > time.time()
        assert states[WATCH].runs == 0
        assert api.texts() == []
    assert task.cancelled() or task.done()


# --- the review-round fixes -------------------------------------------------


def test_an_unchanged_condition_is_not_re_sent(tmp_path: Path) -> None:
    """Review round 1, R1.1: `watch` speaks on CHANGE, not on state.

    Measured before the fix: four ticks of one unchanged crit produced four messages -
    "96 a day for a disk that has not moved", which is how a useful feature gets
    muted. A condition the operator was told about an hour ago is not news; the daily
    line is where standing conditions get repeated.
    """
    store = DigestStore(tmp_path)
    sent = 0
    for tick in range(4):
        run = _run(
            _result("disk", CheckState.CRIT, "/ is 96% full"),
            _result("units", CheckState.OK, "fine"),
            at=NOW + tick * 900,
        )
        digest = render_digest(run, previous=store.last_states(), schedule=WATCH)
        if digest is not None:
            sent += 1
            store.add(digest)
    assert sent == 1, "an unchanged condition was re-sent"

    # A WORSENING is news again.
    worse = _run(
        _result("disk", CheckState.CRIT, "/ is 99% full"),
        _result("units", CheckState.CRIT, "1 unit(s) failed: nginx"),
        at=NOW + 3600,
    )
    digest = render_digest(worse, previous=store.last_states(), schedule=WATCH)
    assert digest is not None
    assert "units" in digest.text
    store.add(digest)

    # And so is a recovery.
    recovered = _run(
        _result("disk", CheckState.CRIT, "/ is 99% full"),
        _result("units", CheckState.OK, "fine"),
        at=NOW + 7200,
    )
    digest = render_digest(recovered, previous=store.last_states(), schedule=WATCH)
    assert digest is not None
    assert "recovered" in digest.text

    # The daily line still repeats the standing condition, because that is its job.
    daily = render_digest(
        _run(_result("disk", CheckState.CRIT, "/ is 99% full"), at=NOW + 7300),
        previous=store.last_states(),
        schedule=DAILY,
        always=True,
    )
    assert daily is not None
    assert "99% full" in daily.text


def test_a_standing_breach_is_not_re_escalated(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """Review round 1, R1.2: one pending collection is the ask; a second is noise.

    Before the fix a breached check proposed on EVERY run, so a full store with
    escalation on meant a new root-action proposal every fifteen minutes, each
    announcing itself, until the helper's per-requester cap started refusing.
    """
    app = _app_with_checks(
        tmp_path,
        helper,
        fake_collector,
        check_escalate_gc=True,
        check_store_dead_paths=1,
        check_disk_warn_percent=0.0,
    )
    client = make_client(app)
    _login(client)

    _drive(client, app, WATCH)
    first = client.get("/api/host/actions").json()
    assert len(first) == 1, "the breach did not propose"

    # Three more passes over the same unchanged breach.
    for _ in range(3):
        _drive(client, app, WATCH)
    assert len(client.get("/api/host/actions").json()) == 1, (
        "a standing breach was re-escalated"
    )
    assert helper.executor.calls == []

    # Once the operator decides it, the queue is free again - and a LATER breach can
    # ask once more, because that is a new ask rather than a repeat.
    action_id = first[0]["proposal"]["id"]
    csrf = client.cookies["scufris_csrf"]
    denied = client.post(
        f"/api/host/actions/{action_id}/deny",
        headers={"Origin": ORIGIN, CSRF_HEADER: csrf},
        json={"reason": "not now"},
    )
    assert denied.status_code == 200, denied.text
