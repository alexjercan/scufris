"""Tests for the reasoning sidecar store.

Each test points ``state_dir`` at a temp dir and exercises the real file I/O -
no codex, no network. The store captures codex "thinking" per (session, turn)
so a hard reload can re-show the spoiler (reasoning is not on disk).
"""

from __future__ import annotations

from pathlib import Path

from scufris.config import Settings
from scufris.reasoning_store import ReasoningStore
from scufris.sessions import reasoning_fingerprint


def _store(tmp_path: Path) -> ReasoningStore:
    return ReasoningStore(Settings(state_dir=tmp_path, _env_file=None))  # type: ignore[call-arg]


def test_append_then_read_roundtrips_in_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append("sess-1", "first thought", answer="answer one")
    store.append("sess-1", "second thought", answer="answer two")

    entries = store.read("sess-1")
    assert [e.reasoning for e in entries] == ["first thought", "second thought"]
    # The answer is stored as its alignment fingerprint, not the raw text.
    assert entries[0].answer == reasoning_fingerprint("answer one")


def test_sessions_are_isolated_in_separate_files(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append("sess-a", "a-think", answer="a")
    store.append("sess-b", "b-think", answer="b")

    assert [e.reasoning for e in store.read("sess-a")] == ["a-think"]
    assert [e.reasoning for e in store.read("sess-b")] == ["b-think"]
    # One file per session under a dedicated reasoning dir.
    files = sorted(p.name for p in (tmp_path / "reasoning").glob("*.json"))
    assert files == ["sess-a.json", "sess-b.json"]


def test_empty_reasoning_still_records_an_entry(tmp_path: Path) -> None:
    # A turn with no thinking still gets an entry (empty), to keep the sidecar
    # 1:1 with the assistant messages the transcript surfaces.
    store = _store(tmp_path)
    store.append("sess-1", "", answer="just an answer")
    entries = store.read("sess-1")
    assert len(entries) == 1
    assert entries[0].reasoning == ""


def test_read_unknown_session_is_empty(tmp_path: Path) -> None:
    assert _store(tmp_path).read("never-seen") == []
    assert _store(tmp_path).read(None) == []


def test_unsafe_session_id_is_a_noop_not_a_traversal(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # A path-traversal id must neither write outside the reasoning dir nor raise.
    store.append("../escape", "evil", answer="x")
    store.append("a/b", "evil", answer="x")
    assert store.read("../escape") == []
    assert not (tmp_path / "escape.json").exists()
    assert not (tmp_path / "reasoning").exists()


def test_tolerant_of_corrupt_sidecar(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append("sess-1", "good", answer="a")
    path = tmp_path / "reasoning" / "sess-1.json"
    path.write_text("{not valid json")
    # A corrupt sidecar reads as empty (degrades to no reasoning), never raises.
    assert store.read("sess-1") == []


def test_ignores_malformed_entries(tmp_path: Path) -> None:
    store = _store(tmp_path)
    (tmp_path / "reasoning").mkdir(parents=True)
    (tmp_path / "reasoning" / "sess-1.json").write_text(
        '{"turns": [{"answer": "f", "reasoning": "ok"}, {"answer": 5}, "junk"]}'
    )
    entries = store.read("sess-1")
    assert [e.reasoning for e in entries] == ["ok"]
