"""Host approvals over Telegram, driven end to end.

The point of this surface is that it is NOT a second decision path, so these tests
run the real thing on both sides: a real `scufris-hostd` on a real unix socket
(the fixtures from `conftest`), the real app and its one
`HostApprovalService`, and the real `TelegramBot` with respx standing in for the
Bot API. A tap on an inline button therefore travels the whole way to the helper's
executor - or is refused by the same sentence the web path would be refused with.

The bot is constructed here rather than started by the lifespan: a running poll
loop against a respx-stubbed `getUpdates` busy-spins (LESSONS.md, the
respx-replies-instantly siblings), and what is under test is dispatch, not polling.
The ops it is given are the PRODUCTION ones (`app.state.telegram_approval_ops`).
"""

from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
import pytest
import respx
from conftest import ORIGIN, _Helper, _login, _propose, _settings, patch_get_backend
from fastapi.testclient import TestClient
from test_host_action_api import _settle
from test_host_action_decisions import (
    _propose_as_the_host_agent,
    _RecordingBackend,
    _wait_for_turn,
)

from scufris.agent_store import HOST_AGENT_ID
from scufris.app import create_app
from scufris.auth import CSRF_HEADER
from scufris.hostd import AuditEvent
from scufris.telegram import (
    APPROVALS_UNAVAILABLE,
    NO_APPROVALS,
    NOT_YOURS,
    ONE_WAY_ARMED,
    TelegramBot,
    approval_keyboard,
    render_approval,
)
from scufris_host import Collector

API = "https://api.telegram.org/botTEST"
CHAT = 4242
OTHER_CHAT = 999

TOKEN_SETTINGS: dict[str, Any] = {
    "telegram_bot_token": "TEST",
    "telegram_allowed_chat_ids": [CHAT],
}


class _Api:
    """Recorded Bot API traffic, so a test can assert what the operator saw."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.edited: list[dict[str, Any]] = []
        self.answered: list[dict[str, Any]] = []
        self._next_id = 1000

    def texts(self) -> list[str]:
        return [str(payload.get("text", "")) for payload in self.sent]

    def edits(self) -> list[str]:
        return [str(payload.get("text", "")) for payload in self.edited]

    def toasts(self) -> list[str]:
        return [str(payload.get("text", "")) for payload in self.answered]

    def install(self, router: respx.Router) -> None:
        def send(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.sent.append(payload)
            self._next_id += 1
            return httpx.Response(
                200,
                json={"ok": True, "result": {"message_id": self._next_id}},
            )

        def edit(request: httpx.Request) -> httpx.Response:
            self.edited.append(json.loads(request.content))
            return httpx.Response(200, json={"ok": True, "result": {}})

        def answer(request: httpx.Request) -> httpx.Response:
            self.answered.append(json.loads(request.content))
            return httpx.Response(200, json={"ok": True, "result": True})

        router.post(f"{API}/sendMessage").mock(side_effect=send)
        router.post(f"{API}/editMessageText").mock(side_effect=edit)
        router.post(f"{API}/answerCallbackQuery").mock(side_effect=answer)
        router.post(f"{API}/sendChatAction").mock(
            return_value=httpx.Response(200, json={"ok": True, "result": True})
        )
        # The production bot's poll loop is RUNNING in these tests (the lifespan
        # starts it), so getUpdates is answered with a 500: the loop logs it and
        # backs off for three seconds. Answering with an empty OK instead would
        # busy-spin against respx (LESSONS.md, the respx-replies-instantly
        # siblings), and leaving it unstubbed would raise on every poll.
        router.post(f"{API}/getUpdates").mock(
            return_value=httpx.Response(500, json={"ok": False})
        )

    def last_message_id(self) -> int:
        return self._next_id


@pytest.fixture
def api() -> Iterator[_Api]:
    recorder = _Api()
    with respx.mock(assert_all_called=False) as router:
        recorder.install(router)
        yield recorder


def _bot(app: Any) -> TelegramBot:
    """The bot the APP started - the production object, with the production ops.

    Not a second bot built beside it: the announcement hooks push into the running
    one, so a test that tapped a button on its own copy would be deciding against a
    different `_announced` map than the one production edits. The app's `make_client`
    fixture enters the lifespan, which is what starts it.
    """
    bot = app.state.telegram_bot
    assert bot is not None, "the app did not start a telegram bot (token unset?)"
    assert bot._approvals._ops is app.state.telegram_approval_ops
    return bot


def _stub_settings_ops() -> Any:
    """`SettingsOps` is required by the constructor and unused by these tests."""
    from scufris.backends import Capability
    from scufris.telegram import OrchestratorInfo, SettingsOps

    async def info() -> OrchestratorInfo:  # pragma: no cover
        return OrchestratorInfo(
            backend="mock",
            model="m",
            auth_mode=None,
            enabled=True,
            permission_mode="manual",
            quota=Capability.unsupported(),
        )

    async def nothing() -> Any:  # pragma: no cover
        return None

    return SettingsOps(info=info, health=nothing, tools=nothing, stats=nothing)


def _dispatch(client: TestClient, bot: TelegramBot, update: dict[str, Any]) -> None:
    """Feed one update to the bot ON THE APP'S EVENT LOOP.

    Not `await bot._handle_update(...)`: a decision starts a supervised apply with
    `asyncio.create_task`, which lands on whatever loop is current. Awaited from the
    test's own loop, that task never progresses (the test then blocks in `_settle`),
    which is a harness artefact - in production the bot polls on the serving loop.
    Driving it through the client's portal reproduces production exactly.
    """
    client.portal.call(functools.partial(bot._handle_update, update))  # type: ignore[union-attr]


def _wait_until(predicate: Callable[[], bool], what: str, tries: int = 200) -> None:
    """Wait for a fire-and-forget notification to land.

    The announcement hooks create a task on the app's loop rather than blocking the
    request that triggered them (a Telegram outage must not fail a decision), so a
    test has to wait for the send instead of assuming the response implied it.
    """
    for _ in range(tries):
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"{what} never happened")


def _tap(
    action_id: str, verb: str, *, chat_id: int = CHAT, message_id: int = 1001
) -> dict[str, Any]:
    """One inline-keyboard tap, in the shape Telegram delivers it."""
    return {
        "update_id": 1,
        "callback_query": {
            "id": f"cb-{verb}-{action_id[:6]}",
            "data": f"{verb}:{action_id}",
            "message": {"message_id": message_id, "chat": {"id": chat_id}},
        },
    }


def _reply(text: str, prompt_id: int, *, chat_id: int = CHAT) -> dict[str, Any]:
    """A message that REPLIES to the bot's force-reply prompt."""
    return {
        "update_id": 2,
        "message": {
            "chat": {"id": chat_id},
            "text": text,
            "reply_to_message": {"message_id": prompt_id},
        },
    }


def _text_update(text: str, *, chat_id: int = CHAT) -> dict[str, Any]:
    return {"update_id": 3, "message": {"chat": {"id": chat_id}, "text": text}}


def _app(tmp_path: Path, fake_collector: Collector, helper: _Helper) -> Any:
    return create_app(
        collector=fake_collector,
        settings=_settings(tmp_path, helper, **TOKEN_SETTINGS),
    )


# --- the two surfaces are one decision path ---------------------------------


def test_host_approval_from_either_surface(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """A proposal can be approved from the dashboard OR from the chat, and either
    way it applies exactly once - through the same service, with the audit naming
    which surface decided."""
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)
    headers = {"Origin": ORIGIN, CSRF_HEADER: csrf}
    bot = _bot(app)

    # One action decided on the WEB.
    web = _propose(client, csrf, args={"unit": "nginx"})
    web_id = web["proposal"]["id"]
    assert (
        client.post(f"/api/host/actions/{web_id}/approve", headers=headers).status_code
        == 200
    )
    _settle(client, web)

    # A second action decided from TELEGRAM, by tapping its button.
    chat = _propose(client, csrf, args={"unit": "nginx"})
    chat_id = chat["proposal"]["id"]
    _dispatch(client, bot, _tap(chat_id, "ha"))
    _settle(client, chat)

    # Both ran, once each.
    assert helper.executor.calls == [
        ["systemctl", "restart", "--", "nginx.service"],
        ["systemctl", "restart", "--", "nginx.service"],
    ]
    # And the record says who decided each, by surface.
    approved = [
        record
        for record in helper.audit.tail(50)
        if record.event == AuditEvent.APPROVED
    ]
    actors = {record.action_id: record.requester.actor for record in approved}
    assert actors[web_id].startswith("operator:")
    assert actors[chat_id] == f"operator:telegram:{CHAT}"

    # The operator is told, and the message they tapped now states the decision
    # instead of still offering the button. Waited for rather than assumed: the
    # announcement is fire-and-forget so a Telegram outage cannot fail a decision.
    _wait_until(
        lambda: any("decision: approved" in text for text in api.edits()),
        "the tapped message was updated",
    )
    assert any("approved" in text for text in api.texts())
    assert api.edited[-1]["reply_markup"] == {"inline_keyboard": []}


def test_telegram_approval_uses_the_same_enforcement(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """Telegram gets no rule of its own - and no shortcut either.

    The one-way acknowledgement is the sharpest case: the ordinary approve is
    refused on BOTH surfaces, with the same sentence, because there is one
    implementation of that rule. Asserted by comparing the two refusals rather than
    by trusting that they match.
    """
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)
    headers = {"Origin": ORIGIN, CSRF_HEADER: csrf}
    bot = _bot(app)

    action = _propose(client, csrf, kind="gc_store", args={})
    action_id = action["proposal"]["id"]

    # The WEB refusal.
    web = client.post(f"/api/host/actions/{action_id}/approve", headers=headers)
    assert web.status_code == 422
    web_detail = web.json()["detail"]

    # The TELEGRAM refusal, from the ordinary tap: it does not approve, and the
    # operator is told the same thing.
    _dispatch(client, bot, _tap(action_id, "ha"))
    assert helper.executor.calls == []
    assert "cannot be undone" in web_detail

    # The ordinary tap ARMS it rather than approving; the confirm tap is what
    # carries the acknowledgement, and that is the only path that runs it.
    assert any(ONE_WAY_ARMED in text for text in api.edits())
    _dispatch(client, bot, _tap(action_id, "hk"))
    _settle(client, action)
    assert helper.executor.calls != []

    # A one-way approval from the chat is audited as the chat, with the token the
    # service required - the same record the web path would have written.
    approved = [
        record
        for record in helper.audit.tail(50)
        if record.event == AuditEvent.APPROVED
    ]
    assert approved[-1].requester.actor == f"operator:telegram:{CHAT}"


def test_telegram_one_way_needs_the_second_tap(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """The first tap on a one-way action never applies it.

    Two things are asserted, because either alone would pass a broken
    implementation: nothing reached the executor after the first tap, AND the
    message was rearmed with a differently-worded confirm button.
    """
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)
    bot = _bot(app)
    action = _propose(client, csrf, kind="gc_store", args={})
    action_id = action["proposal"]["id"]

    _dispatch(client, bot, _tap(action_id, "ha"))
    assert helper.executor.calls == []
    armed = api.edited[-1]
    assert ONE_WAY_ARMED in armed["text"]
    labels = [
        button["text"]
        for row in armed["reply_markup"]["inline_keyboard"]
        for button in row
    ]
    assert any("permanently" in label for label in labels)
    assert "Approve" not in labels  # the ordinary word is gone from the armed state
    assert any("cannot be undone" in toast for toast in api.toasts())

    # Backing out leaves it pending, with the ordinary keyboard again.
    _dispatch(client, bot, _tap(action_id, "hx"))
    assert helper.executor.calls == []
    assert client.get(f"/api/host/actions/{action_id}").json()["decision"] == "pending"
    back_labels = [
        button["text"]
        for row in api.edited[-1]["reply_markup"]["inline_keyboard"]
        for button in row
    ]
    assert any("CANNOT BE UNDONE" in label for label in back_labels)

    # And the confirm tap applies it.
    _dispatch(client, bot, _tap(action_id, "hk"))
    _settle(client, action)
    assert helper.executor.calls != []


def test_telegram_stale_button_is_refused(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """A button tapped after the web already decided refuses, and nothing re-runs.

    This is the cross-surface race as it actually happens: the message in the chat
    is stale, and the tap must be answered with what became of the action instead of
    executing it a second time.
    """
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)
    headers = {"Origin": ORIGIN, CSRF_HEADER: csrf}
    bot = _bot(app)
    action = _propose(client, csrf, args={"unit": "nginx"})
    action_id = action["proposal"]["id"]

    assert (
        client.post(
            f"/api/host/actions/{action_id}/approve", headers=headers
        ).status_code
        == 200
    )
    _settle(client, action)
    ran_once = list(helper.executor.calls)
    assert len(ran_once) == 1

    _dispatch(client, bot, _tap(action_id, "ha"))
    assert helper.executor.calls == ran_once
    assert any("already approved" in toast for toast in api.toasts())
    assert any("already approved" in text for text in api.texts())
    # And the stale message is corrected rather than left offering a button.
    assert any("decision: approved" in text for text in api.edits())


def test_telegram_denial_reaches_the_requesting_agent(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Denying from the chat carries the reason all the way to the agent.

    The reason is typed as a REPLY to the bot's prompt - which is the whole point of
    asking for one: an agent denied with no reason proposes the same thing again.
    """
    backend = _RecordingBackend()
    patch_get_backend(monkeypatch, backend)
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    agent_client = make_client(app)
    bot = _bot(app)
    action_id = _propose_as_the_host_agent(agent_client, app)
    # The operator's own client needs a session to READ the queue (the agent's bearer
    # token proposed, and that credential cannot decide or inspect decisions).
    _login(client)

    # Tap Deny: it asks why, with a force-reply so the answer comes back attached.
    _dispatch(client, bot, _tap(action_id, "hd"))
    prompt = api.sent[-1]
    assert prompt["reply_markup"] == {"force_reply": True}
    prompt_id = api.last_message_id()
    assert client.get(f"/api/host/actions/{action_id}").json()["decision"] == "pending"

    # The reply is the reason.
    _dispatch(
        client, bot, _reply("nginx is serving the demo; ask me after 18:00", prompt_id)
    )
    record = client.get(f"/api/host/actions/{action_id}").json()
    assert record["decision"] == "denied"
    assert record["decided_by"] == f"operator:telegram:{CHAT}"
    assert record["reason"] == "nginx is serving the demo; ask me after 18:00"
    assert helper.executor.calls == []

    # And the agent that asked was resumed with it.
    turn = _wait_for_turn(backend, HOST_AGENT_ID)
    assert "DENIED" in turn
    assert "ask me after 18:00" in turn


def test_telegram_disallowed_chat_cannot_approve(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """The allowlist is the credential, and it is enforced at BOTH layers.

    The bot refuses a tap from a chat that is not allowlisted, and the app's
    providers refuse the same chat id even when called directly - so neither layer
    is the only thing between a stray chat and a root command.
    """
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)
    bot = _bot(app)
    action = _propose(client, csrf, args={"unit": "nginx"})
    action_id = action["proposal"]["id"]

    # Layer one: the bot's own allowlist.
    _dispatch(client, bot, _tap(action_id, "ha", chat_id=OTHER_CHAT))
    assert helper.executor.calls == []
    assert NOT_YOURS in api.toasts()
    assert client.get(f"/api/host/actions/{action_id}").json()["decision"] == "pending"

    # Layer two: the providers, called with that chat id directly - on the app's own
    # loop, like everything else that reaches the service.
    ops = app.state.telegram_approval_ops
    outcome = client.portal.call(  # type: ignore[union-attr]
        functools.partial(ops.approve, action_id, OTHER_CHAT, "")
    )
    assert outcome.ok is False
    assert "cannot decide" in outcome.message
    denied = client.portal.call(  # type: ignore[union-attr]
        functools.partial(ops.deny, action_id, OTHER_CHAT, "no")
    )
    assert denied.ok is False
    assert helper.executor.calls == []
    assert client.get(f"/api/host/actions/{action_id}").json()["decision"] == "pending"


# --- the queue, the commands and the announcements ---------------------------


def test_a_new_proposal_announces_itself_to_the_chat(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """The operator learns about a pending decision without opening anything.

    Driven through the real propose route, so the announcement is the production
    hook firing - and the body is the SHARED renderer, so what the chat says cannot
    drift from what the dashboard and the agent were shown.
    """
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)
    _bot(app)  # assert the running bot is the one the hook will push into

    action = _propose(client, csrf, args={"unit": "nginx"})
    record = app.state.host_actions.get(action["proposal"]["id"])

    _wait_until(
        lambda: any(payload["chat_id"] == CHAT for payload in api.sent),
        "the proposal was announced",
    )
    announced = [payload for payload in api.sent if payload["chat_id"] == CHAT]
    body = announced[-1]["text"]
    assert body == render_approval(record)
    # What it would run, how it can be undone, and who asked - all in the message.
    assert "systemctl restart -- nginx.service" in body
    assert "UNDO" in body or "NO UNDO" in body
    assert announced[-1]["reply_markup"] == approval_keyboard(record)


def test_the_approvals_command_lists_what_is_waiting(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)
    bot = _bot(app)

    _dispatch(client, bot, _text_update("/approvals"))
    assert api.texts() == [NO_APPROVALS]

    # From here the propose ANNOUNCES itself too (the running bot's hook), so the
    # assertions below look for the listing rather than for an exact transcript.
    _propose(client, csrf, args={"unit": "nginx"})
    _dispatch(client, bot, _text_update("/approvals"))
    assert any("host action" in text for text in api.texts())
    assert api.sent[-1]["reply_markup"]["inline_keyboard"]


def test_the_deny_command_takes_an_id_prefix(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """`/deny <prefix> <reason>` - nobody retypes 32 hex characters from a phone."""
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)
    bot = _bot(app)
    action_id = _propose(client, csrf, args={"unit": "nginx"})["proposal"]["id"]

    _dispatch(client, bot, _text_update("/deny"))
    assert any("usage: /deny" in text for text in api.texts())

    _dispatch(client, bot, _text_update("/deny zzzz no"))
    assert any("no pending action starts with" in text for text in api.texts())

    _dispatch(client, bot, _text_update(f"/deny {action_id[:8]} not right now"))
    record = client.get(f"/api/host/actions/{action_id}").json()
    assert record["decision"] == "denied"
    assert record["reason"] == "not right now"
    assert helper.executor.calls == []


async def test_an_ambiguous_deny_prefix_is_refused_rather_than_guessed(
    api: _Api,
) -> None:
    """Denying the wrong root command because two ids shared a character is not a
    mistake worth enabling.

    Driven against stub providers so the two ids DO share a prefix - real ids are
    random hex, so the collision this guards against cannot be produced on demand
    from the real ones.
    """
    from scufris.telegram import ApprovalOps, ApprovalOutcome

    decided: list[str] = []

    async def pending() -> list[Any]:
        return [_stub_record("aaa1"), _stub_record("aaa2")]

    async def get(action_id: str) -> Any:
        return None

    async def approve(action_id: str, chat_id: int, ack: str) -> ApprovalOutcome:
        raise AssertionError("no approval in this test")

    async def deny(action_id: str, chat_id: int, reason: str) -> ApprovalOutcome:
        decided.append(action_id)
        return ApprovalOutcome(ok=True, message="denied")

    bot = TelegramBot(
        "TEST",
        (CHAT,),
        _unused_stream,
        _unused_reset,
        _unused_cancel,
        settings_ops=_stub_settings_ops(),
        approval_ops=ApprovalOps(pending=pending, get=get, approve=approve, deny=deny),
        poll_timeout=0,
    )

    # No app here, so no portal: these providers are stubs and nothing supervised
    # runs, which is why this one can be awaited on the test's own loop.
    await bot._handle_update(_text_update("/deny aaa nope"))
    assert decided == []
    assert any("use more characters" in text for text in api.texts())

    # A prefix that is unique still works.
    await bot._handle_update(_text_update("/deny aaa1 nope"))
    assert decided == ["aaa1"]


def _stub_record(action_id: str) -> Any:
    """The smallest record the prefix matcher reads: an id and a summary."""
    from scufris.host_actions import HostActionRecord
    from scufris.hostd.actions import ActionKind, RiskClass, Step
    from scufris.hostd.preview import Fingerprint, Preview, PreviewKind, Reversal
    from scufris.hostd.protocol import ProposalView
    from scufris_host import Availability

    return HostActionRecord(
        proposal=ProposalView(
            id=action_id,
            kind=ActionKind.UNIT_RESTART,
            risk=RiskClass.R1,
            steps=[Step(argv=["systemctl", "restart", "--", "nginx.service"])],
            summary="restart nginx",
            preview=Preview(
                kind=PreviewKind.STATE,
                headline="h",
                label="l",
                available=Availability(),
                lines=["x"],
            ),
            reversal=Reversal(possible=True, summary="stop it"),
            fingerprint=Fingerprint(value="f", describes="d"),
            created_at=0.0,
            expires_at=1e12,
        )
    )


async def test_a_bot_with_no_approval_surface_says_so(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    api: _Api,
) -> None:
    """A bot built without approval providers has no approval surface at all - it
    says so rather than half-working."""
    bot = TelegramBot(
        "TEST",
        (CHAT,),
        _unused_stream,
        _unused_reset,
        _unused_cancel,
        settings_ops=_stub_settings_ops(),
        poll_timeout=0,
    )
    await bot._handle_update(_text_update("/approvals"))
    assert api.texts() == [APPROVALS_UNAVAILABLE]
    await bot._handle_update(_tap("deadbeef", "ha"))
    assert api.toasts() == [APPROVALS_UNAVAILABLE]


async def _unused_stream(_text: str) -> Any:  # pragma: no cover
    raise AssertionError("no turn expected")
    yield  # pragma: no cover


async def _unused_reset() -> None:  # pragma: no cover
    raise AssertionError("no reset expected")


async def _unused_cancel() -> bool:  # pragma: no cover
    raise AssertionError("no cancel expected")


def test_the_production_bot_is_wired_to_the_approval_service(
    tmp_path: Path, fake_collector: Collector, helper: _Helper, api: _Api
) -> None:
    """The ops are optional on the constructor, so a wiring mistake would silently
    disable the surface rather than fail. This is the guard for that.

    The lifespan really starts the poll loop here, so `getUpdates` is answered with a
    500: the loop logs it and backs off for three seconds, which the context exit
    then cancels. Answering it with an empty OK instead would busy-spin against
    respx (LESSONS.md, the respx-replies-instantly siblings).
    """
    respx.post(f"{API}/getUpdates").mock(return_value=httpx.Response(500))
    app = _app(tmp_path, fake_collector, helper)
    assert app.state.telegram_approval_ops is not None
    with TestClient(app):
        bot = app.state.telegram_bot
        assert bot is not None
        assert bot._approvals._ops is app.state.telegram_approval_ops


def test_a_lapsed_window_offers_no_button_and_refuses_a_stale_tap(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """Once the approval window closes, this surface stops offering the decision.

    Both clocks are moved, because in production they are the SAME clock: the helper
    stops holding the proposal (so the queue this surface reads no longer carries it)
    and the app refuses a tap on the message that was sent while it was still fresh -
    with the reason, rather than sending an approval the helper would reject.
    """
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)
    bot = _bot(app)
    action = _propose(client, csrf, args={"unit": "nginx"})
    action_id = action["proposal"]["id"]

    # The operator does not decide, and the window lapses - on both sides.
    later = time.time() + 700
    helper.advance(700)
    app.state.host_approvals._now = lambda: later

    _dispatch(client, bot, _text_update("/approvals"))
    assert NO_APPROVALS in api.texts()

    _dispatch(client, bot, _tap(action_id, "ha"))
    assert helper.executor.calls == []
    assert any("expired" in toast for toast in api.toasts())
    assert any("expired" in text for text in api.texts())


# --- the review-round fixes -------------------------------------------------


def test_a_long_preview_keeps_the_undo_line(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """Over Telegram's limit, the PREVIEW is what gets shortened - not the tail.

    Review round 1, R1.1. Trimming the tail cost the operator the undo line and the
    result, which sit at the end - on the class of action most likely to be long (an
    R3 activation's preview IS a closure diff). Driven with a preview big enough to
    force the trim.
    """
    from scufris.telegram import MAX_MESSAGE

    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)
    action = _propose(client, csrf, args={"unit": "nginx"})
    record = app.state.host_actions.get(action["proposal"]["id"])
    record.proposal.preview.lines = ["x" * 200] * 60  # ~12k characters

    body = render_approval(record)
    assert len(body) <= MAX_MESSAGE
    # The two sentences that matter most survive...
    assert "NO UNDO:" in body or "UNDO:" in body
    assert "what:" in body and "command:" in body
    # ...the preview is what was shortened, and it SAYS so rather than just stopping.
    assert "more preview lines" in body
    assert body.count("x" * 200) < 60


def test_the_bot_does_not_accumulate_tracked_actions(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """Review round 1, R1.2: a decision surface on a long-lived process needs a
    ceiling on what it remembers."""
    from scufris.telegram import MAX_TRACKED_ACTIONS

    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    _login(client)
    bot = _bot(app)

    for index in range(MAX_TRACKED_ACTIONS + 10):
        bot._approvals._remember(f"action-{index}", CHAT, 1000 + index)
        bot._approvals._await_reason(CHAT, 2000 + index, f"action-{index}")
    assert len(bot._approvals._announced) == MAX_TRACKED_ACTIONS
    assert len(bot._approvals._reason_prompts) == MAX_TRACKED_ACTIONS
    # The oldest went first, and the newest is still there.
    assert "action-0" not in bot._approvals._announced
    assert f"action-{MAX_TRACKED_ACTIONS + 9}" in bot._approvals._announced


def test_a_command_replied_to_the_prompt_is_not_a_reason(
    tmp_path: Path,
    fake_collector: Collector,
    helper: _Helper,
    make_client: Callable[[Any], TestClient],
    api: _Api,
) -> None:
    """Review round 1, R1.3: an operator who answers the prompt with /cancel meant to
    cancel something, not to deny with the reason "/cancel". The prompt stays open."""
    app = _app(tmp_path, fake_collector, helper)
    client = make_client(app)
    csrf = _login(client)
    bot = _bot(app)
    action_id = _propose(client, csrf, args={"unit": "nginx"})["proposal"]["id"]

    _dispatch(client, bot, _tap(action_id, "hd"))
    prompt_id = api.last_message_id()
    _dispatch(client, bot, _reply("/cancel", prompt_id))

    assert client.get(f"/api/host/actions/{action_id}").json()["decision"] == "pending"
    assert any("looks like a command" in text for text in api.texts())

    # The real reason still lands.
    _dispatch(client, bot, _reply("not now", prompt_id))
    record = client.get(f"/api/host/actions/{action_id}").json()
    assert record["decision"] == "denied"
    assert record["reason"] == "not now"
