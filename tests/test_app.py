"""Tests for the FastAPI app: the stats API and static dashboard serving."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scufris.agent import AgentReply, TokenUsage, ToolCall
from scufris.app import create_app
from scufris.config import Settings
from scufris.metrics import Collector
from scufris.processes import ProcessGroup, ProcessInstance, ProcessList


class FakeProcessCollector:
    def sample(self) -> ProcessList:
        return ProcessList(
            groups=[
                ProcessGroup(
                    name="firefox",
                    count=2,
                    cpu_percent=30.0,
                    mem_rss=300,
                    instances=[
                        ProcessInstance(
                            pid=1,
                            username="alex",
                            cpu_percent=20.0,
                            mem_rss=200,
                            num_threads=8,
                            status="running",
                        )
                    ],
                )
            ],
            total=2,
        )


class FakeAgent:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.resets = 0

    async def chat(self, prompt: str) -> AgentReply:
        self.messages.append(prompt)
        return AgentReply(
            text=f"reply: {prompt}",
            status="completed",
            tool_calls=[
                ToolCall(server="scufris", tool="host_stats", status="completed")
            ],
            usage=TokenUsage(input_tokens=120, output_tokens=8),
        )

    def reset(self) -> None:
        self.resets += 1

    async def aclose(self) -> None:
        return None


def _settings(web_dist: Path) -> Settings:
    return Settings(web_dist=web_dist)


def test_api_stats_returns_snapshot(fake_collector: Collector, tmp_path: Path) -> None:
    app = create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))
    client = TestClient(app)

    resp = client.get("/api/stats")
    assert resp.status_code == 200

    body = resp.json()
    assert body["hostname"] == "testbox"
    assert body["mem"]["percent"] == 40.0
    assert body["disks"][0]["mountpoint"] == "/"


def test_api_processes_returns_groups(
    fake_collector: Collector, tmp_path: Path
) -> None:
    app = create_app(
        collector=fake_collector,
        settings=_settings(tmp_path / "absent"),
        process_collector=FakeProcessCollector(),
    )
    resp = TestClient(app).get("/api/processes")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert body["groups"][0]["name"] == "firefox"
    assert body["groups"][0]["instances"][0]["pid"] == 1


def test_api_config_exposes_poll_interval(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(web_dist=tmp_path / "absent", poll_seconds=5.0)
    client = TestClient(create_app(collector=fake_collector, settings=settings))

    resp = client.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["poll_seconds"] == 5.0
    assert body["agent_enabled"] is False


def test_chat_returns_agent_reply(fake_collector: Collector, tmp_path: Path) -> None:
    agent = FakeAgent()
    app = create_app(
        collector=fake_collector, settings=_settings(tmp_path / "absent"), agent=agent
    )
    client = TestClient(app)

    resp = client.post("/api/chat", json={"message": "hello agent"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "reply: hello agent"
    assert agent.messages == ["hello agent"]
    # Per-turn metadata rides along with the reply.
    assert body["tool_calls"][0]["tool"] == "host_stats"
    assert body["usage"]["input_tokens"] == 120


def test_agent_info_reports_model_and_state(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(
        web_dist=tmp_path / "absent",
        agent_model="gpt-5.5",
        agent_enabled=True,
    )
    client = TestClient(create_app(collector=fake_collector, settings=settings))
    body = client.get("/api/agent/info").json()
    assert body["model"] == "gpt-5.5"
    assert body["enabled"] is True
    assert body["auth_mode"] == "chatgpt"


def test_agent_tools_lists_the_mcp_tools(
    fake_collector: Collector, tmp_path: Path
) -> None:
    client = TestClient(
        create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))
    )
    body = client.get("/api/agent/tools").json()
    names = {t["name"] for t in body}
    assert {"host_stats", "tatr_ls", "tatr_show"} <= names
    assert all(t["description"] for t in body)


def test_chat_reset_resets_agent(fake_collector: Collector, tmp_path: Path) -> None:
    agent = FakeAgent()
    app = create_app(
        collector=fake_collector, settings=_settings(tmp_path / "absent"), agent=agent
    )
    client = TestClient(app)

    resp = client.post("/api/chat/reset")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert agent.resets == 1


def test_chat_returns_503_when_agent_disabled(
    fake_collector: Collector, tmp_path: Path
) -> None:
    # Default agent (built from settings) is disabled, so chat is unavailable.
    app = create_app(collector=fake_collector, settings=_settings(tmp_path / "absent"))
    client = TestClient(app)

    resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.status_code == 503
    assert "disabled" in resp.json()["detail"]


def test_index_served_when_dist_exists(
    fake_collector: Collector, tmp_path: Path
) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>scufris</html>")

    app = create_app(collector=fake_collector, settings=_settings(dist))
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    assert "scufris" in resp.text


def test_stats_page_served_at_subpath(
    fake_collector: Collector, tmp_path: Path
) -> None:
    dist = tmp_path / "dist"
    (dist / "stats").mkdir(parents=True)
    (dist / "index.html").write_text("<html>agent</html>")
    (dist / "stats" / "index.html").write_text("<html>stats page</html>")

    app = create_app(collector=fake_collector, settings=_settings(dist))
    client = TestClient(app)

    resp = client.get("/stats/")
    assert resp.status_code == 200
    assert "stats page" in resp.text


def test_api_wins_over_static_mount(fake_collector: Collector, tmp_path: Path) -> None:
    # Even with the static bundle mounted at "/", the API route must resolve.
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>scufris</html>")

    app = create_app(collector=fake_collector, settings=_settings(dist))
    client = TestClient(app)

    resp = client.get("/api/stats")
    assert resp.status_code == 200
    assert resp.json()["hostname"] == "testbox"
