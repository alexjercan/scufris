"""Response models more than one router answers with.

Two of them, and both are shared because the SAME shape is already promised on
more than one path - not because a model looked reusable. Defining a second
`DeleteResult` per router would give pydantic two classes with one name, and the
generated OpenAPI component would be renamed for both.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..sessions import TranscriptMessage


class DeleteResult(BaseModel):
    """What a delete answers: whether it removed anything, and what is current now.

    ``current`` is the session surface's field - deleting the active session
    moves the pointer - and is always None for a project or an agent, which have
    no such pointer. Answered by `/api/projects/{id}`, `/api/agents/{id}` and
    `/api/agent/session/{id}`.
    """

    deleted: bool
    current: str | None


class TranscriptResponse(BaseModel):
    """A conversation's messages, for the two surfaces that replay one: an
    agent's own session (`/api/agents/{id}/transcript`) and one of the
    orchestrator's (`/api/agent/session/{id}`)."""

    messages: list[TranscriptMessage]


__all__ = ["DeleteResult", "TranscriptResponse"]
