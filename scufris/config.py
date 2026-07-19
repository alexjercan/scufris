"""Application settings.

Loaded from the environment (prefix ``SCUFRIS_``) and an optional ``.env`` file
via pydantic-settings. See ``.env.example`` for the knobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root, derived from this file's location: <root>/scufris/config.py.
# In an editable dev install this points at the checkout, so the built frontend
# at <root>/web/dist is found without extra configuration.
_REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCUFRIS_", env_file=".env", extra="ignore"
    )

    host: str = "127.0.0.1"
    port: int = 8000
    # Built dashboard assets to serve at "/". Absent until the frontend is built.
    web_dist: Path = _REPO_ROOT / "web" / "dist"
    # Seconds the frontend waits between /api/stats polls (served to the client).
    poll_seconds: float = 2.0

    # --- Agent (Codex) ---------------------------------------------------
    # Off by default: the agent needs the operator-installed `openai-codex`
    # SDK + codex CLI and a `scufris login`, so a fresh checkout runs the
    # dashboard without it. See tasks/20260719-153040/SPIKE.md.
    agent_enabled: bool = False
    # Model the agent drives (target GPT-5.5; a GPT-5.6 tier if the plan exposes
    # it). Empty string lets Codex pick its default.
    agent_model: str = "gpt-5.5"
    # "chatgpt" = Sign in with ChatGPT subscription (device-code, primary);
    # "api_key" = metered API key fallback.
    agent_auth_mode: Literal["chatgpt", "api_key"] = "chatgpt"
    # API key for the api_key auth mode (SCUFRIS_OPENAI_API_KEY).
    openai_api_key: str | None = None
    # Optional CODEX_HOME override where Codex stores its auth/session state.
    codex_home: Path | None = None
