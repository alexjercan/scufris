"""Tests for the FastAPI app: the stats API and static dashboard serving."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scufris.app import create_app
from scufris.config import Settings
from scufris.metrics import Collector


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


def test_api_config_exposes_poll_interval(
    fake_collector: Collector, tmp_path: Path
) -> None:
    settings = Settings(web_dist=tmp_path / "absent", poll_seconds=5.0)
    client = TestClient(create_app(collector=fake_collector, settings=settings))

    resp = client.get("/api/config")
    assert resp.status_code == 200
    assert resp.json()["poll_seconds"] == 5.0


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
