"""What a configuration change is, and what a build publishes while it runs.

The build gets its own event type rather than reusing the host-apply one: a
build log is not model text and a built configuration is not an applied action.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Callable, Literal

from pydantic import BaseModel, Field

from ..eventbus import EventBus
from ..supervisor import Supervisor


class ConfigChangeRefused(Exception):
    """Something about the request or the repository makes a build impossible.

    Always carries a sentence an operator can act on - "that ref does not exist
    in that repository", not "git failed".
    """


class ChangeState(StrEnum):
    """Where a configuration change is in its life.

    ``FAILED`` and ``CANCELLED`` are terminal and carry no proposal: a change
    that did not build cannot be approved, because there is nothing to activate.
    """

    BUILDING = "building"
    PROPOSED = "proposed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Resolved(BaseModel):
    """What a ref means right now, and what the operator should know about it."""

    repo: str
    ref: str
    rev: str
    subject: str = ""
    # Whether the revision is already contained in what the operator's own
    # checkout has. A change built from a branch that is not merged yet is
    # normal and fine - it just means merging it back is still to do, which is a
    # project act and not something this flow does.
    merged: bool | None = None
    head_branch: str = ""
    # Files modified in the working tree the ref was resolved from. They are NOT
    # in the build, by construction, and saying so is the difference between an
    # honest preview and an agent wondering why its edit did nothing.
    uncommitted: list[str] = Field(default_factory=list)


class ConfigChange(BaseModel):
    """One configuration change as the API and the dashboard see it."""

    id: str
    resolved: Resolved
    attr: str
    state: ChangeState = ChangeState.BUILDING
    # The built system, once there is one.
    toplevel: str = ""
    # The host action proposal that carries the activation, once it exists. This
    # is the id every approval, denial and audit record uses.
    action_id: str = ""
    run_id: str = ""
    log_tail: str = ""
    error: str = ""
    created_at: float = 0.0
    # Which agent asked, when one did. Recorded for display; it grants nothing.
    agent: str = ""
    requested_by: str = ""

    @property
    def repo(self) -> str:
        return self.resolved.repo


class ConfigBuildOutput(BaseModel):
    type: Literal["output"] = "output"
    stream: str
    text: str


class ConfigBuildDone(BaseModel):
    type: Literal["done"] = "done"
    change: ConfigChange


class ConfigBuildError(BaseModel):
    type: Literal["error"] = "error"
    detail: str


ConfigBuildEvent = ConfigBuildOutput | ConfigBuildDone | ConfigBuildError

ConfigSupervisor = Supervisor[ConfigBuildEvent]
ConfigBuildBus = EventBus[ConfigBuildEvent]


def _build_error_event(detail: str) -> ConfigBuildEvent:
    return ConfigBuildError(detail=detail)


def _build_error_detail(event: ConfigBuildEvent) -> str | None:
    return event.detail if isinstance(event, ConfigBuildError) else None


def config_supervisor(
    *,
    max_concurrent: int = 1,
    max_history: int = 50,
    clock: Callable[[], float] = time.time,
) -> ConfigSupervisor:
    """A supervisor for configuration builds.

    Separate from the host-apply supervisor on purpose. A NixOS build can run for
    a long time and needs no privilege; sharing the single apply slot with it
    would mean a kernel rebuild blocks an unrelated service restart the operator
    approved.
    """
    return Supervisor(
        error_event=_build_error_event,
        error_detail=_build_error_detail,
        max_concurrent=max_concurrent,
        max_history=max_history,
        clock=clock,
    )
