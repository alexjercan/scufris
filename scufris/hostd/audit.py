"""The append-only record, written by root, that the app cannot rewrite.

From the spike: the privileged half of the audit is appended by the HELPER, to
its own root-owned log, so it survives the app being the thing that misbehaved.
It deliberately does not share the app's transactional store.

Three properties, each enforced by construction rather than by discipline:

- **Append-only.** The only write path is ``append``. There is no update, no
  delete, and no protocol verb reaches this module - a verb that erases the
  record of privileged actions is precisely what an R4 refusal is for.
- **Bounded.** The log rotates on SIZE, not on a timer, so a burst cannot
  outrun the policy. Records are single-line JSON, so a rotation boundary never
  splits one. The oldest rotated file is pruned first; that pruning is the
  retention policy and is the only deletion in the module.
- **Redacted.** Anything secret-shaped is replaced before it is written, by key
  name and by value, so the log itself never becomes the leak.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from .actions import ActionKind, RiskClass

logger = logging.getLogger(__name__)

# The retention knob the spike left to this task. 16 MiB of single-line JSON is
# on the order of 30k records; five files bound the log at 80 MiB, which is
# small next to a nix store and large enough that a rotation is a rare event.
DEFAULT_MAX_BYTES = 16 * 1024 * 1024
DEFAULT_KEEP = 5

# Key names whose VALUE never belongs in a record, matched anywhere in the key.
_SECRET_KEY = re.compile(
    r"secret|token|password|passwd|credential|api_?key|private_?key|cookie",
    re.IGNORECASE,
)

REDACTED = "[redacted]"


class AuditEvent(StrEnum):
    """Every point in an action's life that produces a record.

    ``REFUSED`` and ``DENIED`` are different sentences and are kept apart on
    purpose: REFUSED is the helper declining to build the action at all (an
    unknown verb, a deny-listed unit, an argument that would become a flag),
    DENIED is the operator saying no to something the helper was willing to do.
    """

    REQUESTED = "requested"
    REFUSED = "refused"
    DENIED = "denied"
    APPROVED = "approved"
    APPLIED = "applied"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class Requester(BaseModel):
    """Who asked. An agent proposes; only the operator approves."""

    actor: str = ""
    agent: str = ""
    run: str = ""


class AuditRecord(BaseModel):
    """One line of the log."""

    ts: float
    at: str
    event: AuditEvent
    action_id: str
    kind: ActionKind | None = None
    risk: RiskClass | None = None
    args: dict[str, object] = Field(default_factory=dict)
    argv: list[str] = Field(default_factory=list)
    requester: Requester = Field(default_factory=Requester)
    outcome: str = ""
    returncode: int | None = None
    duration_seconds: float = 0.0
    # What the inverse is, or the recorded reason there is none. Written at
    # apply time so the record itself answers "can this be undone".
    reversal: str = ""
    # The generation or unit state this action can be rolled back to, when the
    # class has one.
    restore_point: str = ""
    detail: str = ""


def redact(value: object, *, secrets: frozenset[str] = frozenset()) -> object:
    """Strip secret-shaped data from ``value`` before it is written.

    Two passes, because they catch different things: a key named ``secret``
    hides its value whatever that value looks like, and a known secret STRING is
    replaced wherever it appears - including inside a message that merely quotes
    it.
    """
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            if _SECRET_KEY.search(name):
                out[name] = REDACTED
            else:
                out[name] = redact(item, secrets=secrets)
        return out
    if isinstance(value, list):
        return [redact(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        cleaned = value
        for secret in secrets:
            if secret and secret in cleaned:
                cleaned = cleaned.replace(secret, REDACTED)
        return cleaned
    return value


class AuditLog:
    """A size-rotated, append-only JSON-lines log.

    Not thread-safe by construction and it does not need to be: the helper is
    single-process and appends from one event loop. The file is opened with
    ``O_APPEND`` per write, so even a concurrent writer cannot interleave inside
    a line.
    """

    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        keep: int = DEFAULT_KEEP,
        secrets: frozenset[str] = frozenset(),
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self.max_bytes = max(1024, int(max_bytes))
        # At least TWO: `keep` counts the active file plus its rotated siblings,
        # and keep=1 would mean the rotation path unlinks the log outright -
        # "delete the whole audit on overflow", reachable from a configuration
        # that merely looked conservative (review round 1, R1.8). One rotated
        # sibling is the minimum that makes rotation a rotation.
        self.keep = max(2, int(keep))
        self._secrets = secrets
        self._now = clock

    # --- writing ---------------------------------------------------------

    def append(self, record: AuditRecord) -> None:
        """Write one record. The only write path there is."""
        payload = redact(record.model_dump(mode="json"), secrets=self._secrets)
        line = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
        encoded = line.encode("utf-8")
        self._rotate_if_needed(len(encoded))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 0600 and O_APPEND: the log is root's, and every write lands at the end
        # of the file even if something else has it open.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, encoded)
        finally:
            os.close(fd)

    def record(self, event: AuditEvent, **fields: object) -> AuditRecord:
        """Build a record stamped with the current time and append it."""
        moment = float(self._now())
        payload: dict[str, object] = {
            "ts": moment,
            "at": datetime.fromtimestamp(moment, tz=timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        entry = AuditRecord.model_validate(payload)
        self.append(entry)
        return entry

    def _rotated(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{index}")

    def _rotate_if_needed(self, incoming: int) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size + incoming <= self.max_bytes:
            return
        # Oldest first: the file that would become .keep is dropped, then each
        # remaining one shifts up, then the active file becomes .1.
        # keep >= 2 is enforced in __init__, so there is always at least one
        # rotated slot and the active file is never simply deleted.
        oldest = self.keep - 1
        self._rotated(oldest).unlink(missing_ok=True)
        for index in range(oldest - 1, 0, -1):
            source = self._rotated(index)
            if source.exists():
                source.rename(self._rotated(index + 1))
        self.path.rename(self._rotated(1))
        logger.info("hostd: rotated the audit log at %d bytes", size)

    # --- reading ---------------------------------------------------------

    def files_newest_first(self) -> list[Path]:
        """The active log then its rotated siblings, newest to oldest."""
        found = [self.path] if self.path.exists() else []
        for index in range(1, self.keep):
            candidate = self._rotated(index)
            if candidate.exists():
                found.append(candidate)
        return found

    def tail(self, limit: int = 50) -> list[AuditRecord]:
        """The most recent ``limit`` records, oldest first.

        Reads backwards through the rotation set and stops as soon as it has
        enough, so a bounded read never depends on the log's total size.
        """
        wanted = max(0, int(limit))
        if wanted == 0:
            return []
        collected: list[AuditRecord] = []
        for path in self.files_newest_first():
            lines = _tail_lines(path, wanted - len(collected))
            parsed = [record for record in (_parse(line) for line in lines) if record]
            collected = parsed + collected
            if len(collected) >= wanted:
                break
        return collected[-wanted:]


# How much of a file's end is read to find its last lines. Comfortably more than
# `limit` records at any realistic record size, and bounded regardless of the
# file.
_TAIL_CHUNK = 1024 * 1024


def _tail_lines(path: Path, limit: int) -> list[str]:
    if limit <= 0:
        return []
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > _TAIL_CHUNK:
                handle.seek(size - _TAIL_CHUNK)
                handle.readline()  # drop the partial line the seek landed in
            data = handle.read()
    except OSError as exc:
        logger.warning("hostd: cannot read audit file %s: %s", path, exc)
        return []
    lines = [
        line for line in data.decode("utf-8", "replace").splitlines() if line.strip()
    ]
    return lines[-limit:]


def _parse(line: str) -> AuditRecord | None:
    try:
        return AuditRecord.model_validate_json(line)
    except ValueError:
        # A line this build cannot read is skipped rather than raised: the log is
        # append-only history and may outlive a schema change.
        logger.info("hostd: skipping an audit line this build cannot parse")
        return None
