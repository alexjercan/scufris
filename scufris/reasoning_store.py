"""Sidecar store for codex reasoning ("thinking") text.

Codex persists reasoning only as an ``encrypted_content`` blob in its rollout,
so the plaintext is unrecoverable from disk (task 20260726-215910). To let a
hard page reload re-show the thinking spoiler that streamed live, scufris
captures the live ``reasoning_delta`` stream per (session, turn) into this
sidecar, and ``CodexBackend.read_transcript`` merges it back into the
transcript.

Layout: ONE JSON file per session under ``<state_dir>/reasoning/<session_id>``
``.json`` - so a turn's append rewrites only that session's reasoning, not a
shared file across all sessions (the other stores hold one small row per agent;
this holds a growing per-turn list, so a shared-file atomic rewrite would be
O(all sessions) per append). See ``tasks/20260726-215910/DECISION.md``.

Shape::

    {"turns": [{"answer": "<fingerprint>", "reasoning": "..."}, ...]}

ordered oldest->newest, one entry per COMPLETED assistant turn (empty
``reasoning`` when the model did not think, to keep the list 1:1 with the
assistant messages ``read_transcript`` surfaces). ``answer`` is a
whitespace-normalized fingerprint of the turn's final answer, used at merge
time to guard sidecar<->transcript alignment (see ``sessions.merge_reasoning``).

Persistence mirrors the other stores' write discipline (atomic tmp+replace,
tolerant load); it records server-internal turn output, not a user config edit,
so it is not gated by ``settings_writable``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config import Settings
from .sessions import reasoning_fingerprint

logger = logging.getLogger(__name__)

# codex session ids are UUID-ish; refuse anything that could escape the reasoning
# dir (path separators, ``..``) before using it as a filename.
_SAFE_SESSION_ID = re.compile(r"[A-Za-z0-9._-]+")


class ReasoningEntry(BaseModel):
    """One completed assistant turn's captured reasoning + its alignment key."""

    answer: str  # fingerprint of the turn's final answer text
    reasoning: str


class ReasoningStore:
    """Per-session sidecar of captured reasoning text, under the state dir."""

    def __init__(self, settings: Settings) -> None:
        self._dir = Path(settings.state_dir) / "reasoning"

    def _path(self, session_id: str | None) -> Path | None:
        """The sidecar path for ``session_id``, or None when the id is missing or
        unsafe as a filename (so a bad id is a no-op, never a traversal)."""
        if not session_id or _SAFE_SESSION_ID.fullmatch(session_id) is None:
            return None
        return self._dir / f"{session_id}.json"

    def append(self, session_id: str | None, reasoning: str, *, answer: str) -> None:
        """Record one completed turn's reasoning, keyed by ``session_id``.

        ``answer`` is the turn's final answer text; it is fingerprinted for
        alignment. A missing/unsafe id, or any I/O error, is swallowed - a lost
        sidecar entry degrades to "no reasoning on reload", never a failed turn.
        """
        path = self._path(session_id)
        if path is None:
            return
        entries = self._load(path)
        entries.append(
            {"answer": reasoning_fingerprint(answer), "reasoning": reasoning}
        )
        self._persist(path, entries)

    def read(self, session_id: str | None) -> list[ReasoningEntry]:
        """The session's captured turns, oldest-first; empty when absent."""
        path = self._path(session_id)
        if path is None:
            return []
        out: list[ReasoningEntry] = []
        for raw in self._load(path):
            answer = raw.get("answer")
            reasoning = raw.get("reasoning")
            if isinstance(answer, str) and isinstance(reasoning, str):
                out.append(ReasoningEntry(answer=answer, reasoning=reasoning))
        return out

    def _load(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("reasoning sidecar: cannot read %s: %s", path, exc)
            return []
        turns = data.get("turns") if isinstance(data, dict) else None
        if not isinstance(turns, list):
            return []
        return [t for t in turns if isinstance(t, dict)]

    def _persist(self, path: Path, entries: list[dict[str, Any]]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"turns": entries}, indent=2))
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("reasoning sidecar: cannot write %s: %s", path, exc)
