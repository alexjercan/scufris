"""The orchestrator's chat router, driven over a fake turn service on a bare app.

A third file in the `test_domain_routers.py` / `test_orchestrator_routers.py`
family, for the same reason the second one exists: the shared file is at its line
cap. The rig pattern is unchanged.

What is worth proving here is the split of responsibility the extraction is FOR:
the turn service owns running a turn, and the router owns the transport around
it - the disabled-agent refusal that must land before anything is decoded, the
base64 attachment, and the shape a decode failure takes on the wire.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scufris.agent import AgentReply, StreamDone
from scufris.api.chat import (
    ChatDeps,
    ChatRequest,
    ImageAttachment,
    build_chat_router,
    write_image_to_temp,
)
from scufris.config import Settings
from scufris.orchestrator import AgentDisabled, RunAlreadyActive
from scufris_core import EventBus

# A 1x1 PNG, as the browser sends it.
PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)


class FakeTurnService:
    """`OrchestratorTurnService`'s answers, recorded. Refusals scripted by name."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.raises: Exception | None = None
        self.image_paths: list[str] | None = None
        self.on_done: Any = None

    async def send(self, message: str) -> AgentReply:
        self.calls.append(("send", message))
        if self.raises is not None:
            raise self.raises
        return AgentReply(text="ok")

    async def stream(
        self, message: str, *, image_paths: Any = None, on_done: Any = None
    ) -> tuple[str, EventBus[Any]]:
        self.calls.append(("stream", message))
        self.image_paths = image_paths
        self.on_done = on_done
        if self.raises is not None:
            raise self.raises
        # A launched turn ends: publish `done` and close, or the SSE relay never
        # finishes and the TestClient request blocks forever.
        bus: EventBus[Any] = EventBus()
        bus.publish(StreamDone(reply=AgentReply(text="ok")))
        bus.close()
        return "run-1", bus

    async def reset(self) -> None:
        self.calls.append(("reset", None))


class ChatRig:
    """The chat router on a bare app, with the turn fake reachable by name."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.turn = FakeTurnService()
        self.app = FastAPI()
        self.app.include_router(
            build_chat_router(ChatDeps(settings=settings, turn=cast(Any, self.turn)))
        )


def _settings(tmp_path: Path, **kwargs: Any) -> Settings:
    return Settings(
        state_dir=tmp_path / "state",
        web_dist=tmp_path / "dist",
        _env_file=None,  # type: ignore[call-arg]
        **kwargs,
    )


@pytest.fixture
def chat_rig(tmp_path: Path) -> ChatRig:
    return ChatRig(_settings(tmp_path))


@pytest.fixture
def chat_client(chat_rig: ChatRig) -> Iterator[TestClient]:
    with TestClient(chat_rig.app) as client:
        yield client


def test_a_chat_turn_is_the_turn_services_answer(
    chat_rig: ChatRig, chat_client: TestClient
) -> None:
    response = chat_client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "ok"
    assert chat_rig.turn.calls == [("send", "hello")]


@pytest.mark.parametrize(
    ("raised", "status"),
    [
        (RunAlreadyActive("already running"), 409),
        (AgentDisabled("agent is disabled"), 503),
    ],
)
def test_the_chat_router_maps_every_orchestrator_refusal(
    chat_rig: ChatRig, chat_client: TestClient, raised: Exception, status: int
) -> None:
    """The service decides; the router only says it in HTTP."""
    chat_rig.turn.raises = raised
    assert chat_client.post("/api/chat", json={"message": "hi"}).status_code == status
    assert (
        chat_client.post("/api/chat/stream", json={"message": "hi"}).status_code
        == status
    )


def test_resetting_forgets_the_conversation_through_the_service(
    chat_rig: ChatRig, chat_client: TestClient
) -> None:
    assert chat_client.post("/api/chat/reset").json() == {"ok": True}
    assert chat_rig.turn.calls == [("reset", None)]


def test_a_disabled_agent_is_refused_before_anything_is_decoded(
    tmp_path: Path,
) -> None:
    """503 ahead of the attachment: a disabled agent answers 503 even when the
    image is ALSO bad, rather than a 200 carrying an error frame."""
    rig = ChatRig(_settings(tmp_path, agent_enabled=False))
    with TestClient(rig.app) as client:
        streamed = client.post(
            "/api/chat/stream",
            json={"message": "hi", "image": {"data_base64": "!!", "mime": "image/png"}},
        )
        assert streamed.status_code == 503, streamed.text
        assert client.post("/api/chat/reset").status_code == 503
    assert not rig.turn.calls


def test_an_attachment_reaches_the_turn_as_a_path_on_disk(
    chat_rig: ChatRig, chat_client: TestClient
) -> None:
    """The service is handed a file, not base64: decoding is this transport's job."""
    response = chat_client.post(
        "/api/chat/stream",
        json={
            "message": "look",
            "image": {"data_base64": PNG_BASE64, "mime": "image/png"},
        },
    )
    assert response.status_code == 200, response.text
    paths = chat_rig.turn.image_paths
    assert paths is not None and len(paths) == 1
    assert Path(paths[0]).suffix == ".png"

    # The tempdir is the TURN's to clean, not the relay's: it survives the
    # response and goes away when the completion callback runs.
    assert Path(paths[0]).is_file()
    chat_rig.turn.on_done()
    assert not Path(paths[0]).exists()


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"data_base64": PNG_BASE64, "mime": "application/pdf"}, "unsupported"),
        ({"data_base64": "not base64!", "mime": "image/png"}, "base64"),
    ],
)
def test_a_bad_attachment_is_one_error_frame_and_no_turn(
    chat_rig: ChatRig, chat_client: TestClient, payload: dict[str, str], detail: str
) -> None:
    """It never launches: a turn started on an undecodable attachment would
    burn a supervised run to say nothing."""
    response = chat_client.post(
        "/api/chat/stream", json={"message": "look", "image": payload}
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [f["kind"] for f in frames] == ["error"]
    assert detail in frames[0]["detail"]
    assert not chat_rig.turn.calls


def test_an_oversize_attachment_is_refused_before_it_is_written(
    tmp_path: Path,
) -> None:
    """The cap is on the DECODED bytes, so a small base64 bomb cannot exhaust
    memory or fill the temp dir."""
    import base64

    huge = base64.b64encode(b"\x00" * (13 * 1024 * 1024)).decode()
    with pytest.raises(ValueError, match="too large"):
        write_image_to_temp(ImageAttachment(data_base64=huge, mime="image/png"))


def test_the_chat_router_reaches_for_nothing(
    chat_rig: ChatRig, chat_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It constructs no `Settings` of its own; the one it answers from is the
    one handed to `ChatDeps`."""

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the chat router constructed a settings object")

    monkeypatch.setattr(Settings, "__init__", _forbidden)
    assert chat_client.post("/api/chat", json={"message": "hi"}).status_code == 200
    assert chat_client.post("/api/chat/reset").status_code == 200


def test_the_chat_request_model_rejects_a_missing_message() -> None:
    """The message is the turn; an absent one is a 422 at the boundary rather
    than an empty prompt handed to a backend."""
    with pytest.raises(ValueError):
        ChatRequest(**{})  # type: ignore[arg-type]
