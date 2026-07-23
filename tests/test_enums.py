"""The shared option enums are StrEnums: membership is validated, but the wire
format (JSON string) and `==` against the raw string are unchanged."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from scufris.config import Settings
from scufris.enums import AgentState, AuthMode, Backend, PermissionMode, RunPhase


def test_members_equal_their_string_and_serialize_as_it() -> None:
    # StrEnum member IS its string: comparisons and json.dumps are unchanged.
    assert AuthMode.API_KEY == "api_key"
    assert Backend.CODEX == "codex"
    assert PermissionMode.MANUAL == "manual"
    assert f"{AgentState.RUNNING}" == "running"  # f-string -> the value, not repr
    assert json.dumps({"k": RunPhase.DONE}) == '{"k": "done"}'


class _M(BaseModel):
    auth: AuthMode
    perm: PermissionMode


def test_pydantic_validates_membership_and_round_trips_on_the_wire() -> None:
    # A valid value round-trips unchanged as the plain string on the wire. The raw
    # str is the input under test here (pydantic coercion), so the arg-type mismatch
    # against the StrEnum field is deliberate.
    m = _M(auth="api_key", perm="edit")  # type: ignore[arg-type]
    assert isinstance(m.auth, AuthMode)
    assert m.model_dump_json() == '{"auth":"api_key","perm":"edit"}'
    # An out-of-set value is rejected by pydantic.
    with pytest.raises(ValidationError):
        _M(auth="nope", perm="edit")  # type: ignore[arg-type]


def test_settings_reject_invalid_backend_and_auth(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCUFRIS_AGENT_AUTH_MODE", "not-a-mode")
    with pytest.raises(ValidationError):
        Settings()
    monkeypatch.delenv("SCUFRIS_AGENT_AUTH_MODE")
    monkeypatch.setenv("SCUFRIS_AGENT_BACKEND", "not-a-backend")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_valid_values_are_enums_that_match_their_string() -> None:
    # Raw strings are the input under test (they must coerce to the StrEnum member),
    # so the arg-type mismatch against the enum-typed fields is deliberate.
    s = Settings(agent_auth_mode="api_key", agent_backend="claude")  # type: ignore[arg-type]
    assert isinstance(s.agent_auth_mode, AuthMode)
    assert s.agent_auth_mode == "api_key"  # wire value unchanged
    assert s.agent_backend == "claude"
    # A legacy codex mode id still coerces to the canonical enum member.
    assert (
        Settings(agent_backend="app_server").agent_backend  # type: ignore[arg-type]
        is Backend.CODEX
    )
