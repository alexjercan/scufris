"""The helper's audit log: bounded, append-only, and never split mid-record.

The log is the thing that has to be trustworthy when the app is not, so the
properties tested here are the ones that would quietly stop being true: that it
cannot grow forever, that rotation drops the OLDEST rather than the newest, that
a rotation boundary never lands inside a record, and that nothing in the module
or the protocol offers a way to erase an entry.
"""

from __future__ import annotations

import json
from pathlib import Path

from scufris_hostd import ActionKind, AuditEvent, AuditLog, Verb
from scufris_hostd.audit import REDACTED, redact


def _fill(log: AuditLog, count: int, *, start: int = 0) -> None:
    for index in range(start, start + count):
        log.record(
            AuditEvent.APPLIED,
            action_id=f"action-{index:05d}",
            kind=ActionKind.UNIT_RESTART,
            argv=["systemctl", "restart", "--", "nginx.service"],
            detail="x" * 200,
        )


def test_audit_log_rotates_and_never_deletes(tmp_path: Path) -> None:
    """It rotates on size, keeps a bounded set, and prunes oldest-first."""
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, max_bytes=4096, keep=3)

    _fill(log, 200)

    assert path.exists()
    assert (tmp_path / "audit.jsonl.1").exists()
    assert (tmp_path / "audit.jsonl.2").exists()
    # keep=3 is the TOTAL, so there is no third rotated file.
    assert not (tmp_path / "audit.jsonl.3").exists()

    for candidate in log.files_newest_first():
        assert candidate.stat().st_size <= 4096 + 1024

    # Oldest-first pruning: the earliest ids are gone, the latest are held.
    surviving = _all_ids(tmp_path)
    assert "action-00199" in surviving
    assert "action-00000" not in surviving


def test_rotation_never_splits_a_record(tmp_path: Path) -> None:
    """Single-line JSON is what makes a rotation boundary safe."""
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, max_bytes=2048, keep=4)
    _fill(log, 150)

    for candidate in log.files_newest_first():
        for line in candidate.read_text().splitlines():
            if line.strip():
                parsed = json.loads(line)
                assert parsed["action_id"].startswith("action-")


def test_tail_reads_across_a_rotation_boundary_oldest_first(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    # Sized so every record survives: this test is about the READ spanning a
    # rotation boundary, not about pruning (which has its own test above).
    log = AuditLog(path, max_bytes=4096, keep=10)
    _fill(log, 60)

    records = log.tail(30)

    assert len(records) == 30
    ids = [record.action_id for record in records]
    assert ids == sorted(ids)  # oldest first
    assert ids[-1] == "action-00059"
    # The tail spans more than the active file, which is far smaller than 30
    # records at this size.
    assert len(log.files_newest_first()) > 1


def test_tail_of_an_empty_log_is_empty_not_an_error(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", max_bytes=4096, keep=3)
    assert log.tail(10) == []


def test_a_line_this_build_cannot_parse_is_skipped_not_fatal(tmp_path: Path) -> None:
    """The log is append-only history and may outlive a schema change."""
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, max_bytes=1 << 20, keep=3)
    _fill(log, 2)
    with path.open("a") as handle:
        handle.write('{"from":"a future version"}\n')
    _fill(log, 1, start=99)

    records = log.tail(10)
    assert [record.action_id for record in records] == [
        "action-00000",
        "action-00001",
        "action-00099",
    ]


def test_the_audit_offers_no_way_to_delete_an_entry() -> None:
    """A verb that erases the record of privileged actions is what R4 is for.

    Pruning the OLDEST rotated file is the retention policy and lives inside the
    log; there is no API for removing a record, and no protocol verb reaches the
    log at all except to read a bounded tail.
    """
    api = {name for name in dir(AuditLog) if not name.startswith("_")}
    assert api == {"append", "record", "tail", "files_newest_first"}

    verbs = {verb.value for verb in Verb}
    assert not {verb for verb in verbs if "delete" in verb or "prune" in verb}
    assert "audit_tail" in verbs


def test_redaction_hides_a_secret_by_key_and_by_value() -> None:
    payload = {
        "secret": "hunter2",
        "api_key": "abc",
        "detail": "the caller sent token-xyz on the socket",
        "argv": ["systemctl", "restart", "--", "nginx.service"],
        "nested": [{"password": "p"}, {"fine": "value"}],
    }

    cleaned = redact(payload, secrets=frozenset({"token-xyz"}))

    assert cleaned == {
        "secret": REDACTED,
        "api_key": REDACTED,
        "detail": f"the caller sent {REDACTED} on the socket",
        "argv": ["systemctl", "restart", "--", "nginx.service"],
        "nested": [{"password": REDACTED}, {"fine": "value"}],
    }


def _all_ids(directory: Path) -> set[str]:
    found: set[str] = set()
    for candidate in directory.glob("audit.jsonl*"):
        for line in candidate.read_text().splitlines():
            if line.strip():
                found.add(json.loads(line)["action_id"])
    return found


def test_a_refusal_flood_cannot_rotate_the_history_away(tmp_path: Path) -> None:
    """The record must survive the caller most likely to want it gone.

    Reaching the socket needs no secret, so an unauthenticated caller can
    produce refusals as fast as it can open connections - measured at ~4000/s in
    review round 1, which erased a genuine record from a small log and would
    clear the production 80 MiB in about ninety seconds. Coalescing is what
    makes "nothing outside the helper can delete an entry" true in effect and
    not just on paper (R1.2).
    """
    from scufris_hostd import HostdEngine
    from scufris_hostd.engine import REFUSAL_WINDOW_SECONDS

    moment = [1000.0]
    log = AuditLog(
        tmp_path / "audit.jsonl", max_bytes=4096, keep=2, clock=lambda: moment[0]
    )
    engine = HostdEngine(log, clock=lambda: moment[0])

    log.record(AuditEvent.APPLIED, action_id="REAL-HISTORY-ENTRY")

    for _ in range(5000):
        engine.record_refusal("a caller presented no valid secret")

    surviving = _all_ids(tmp_path)
    assert "REAL-HISTORY-ENTRY" in surviving, (
        "a refusal flood rotated the genuine history off disk"
    )

    # The operator still learns it happened - the first refusal is written
    # immediately, because "someone is trying" is the signal that matters.
    refusals = [r for r in log.tail(50) if r.event is AuditEvent.REFUSED]
    assert len(refusals) == 1

    # And the count is not silently lost: closing the window leaves a summary.
    moment[0] += REFUSAL_WINDOW_SECONDS + 1
    engine.record_refusal("a caller presented no valid secret")
    details = [r.detail for r in log.tail(50) if r.event is AuditEvent.REFUSED]
    assert any("4999 further refusals" in detail for detail in details), details


def test_a_different_refusal_is_still_recorded_during_a_flood(tmp_path: Path) -> None:
    """Coalescing must not swallow a NEW kind of refusal.

    The paired guard for the test above: a rate limit that hides everything
    while an attacker floods would trade one failure for a worse one.
    """
    from scufris_hostd import HostdEngine

    log = AuditLog(tmp_path / "audit.jsonl", max_bytes=1 << 20, keep=3)
    engine = HostdEngine(log)

    for _ in range(100):
        engine.record_refusal("a caller presented no valid secret")
    engine.record_refusal("something else entirely")

    details = [r.detail for r in log.tail(50) if r.event is AuditEvent.REFUSED]
    assert "a caller presented no valid secret" in details
    assert "something else entirely" in details
