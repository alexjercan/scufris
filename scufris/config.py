"""Application settings.

Loaded from the environment (prefix ``SCUFRIS_``) and an optional ``.env`` file
via pydantic-settings. See ``.env.example`` for the knobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class McpServerSpec(BaseModel):
    """An extra MCP server to register with codex, beyond the built-in Scufris one.

    Declared in config (e.g. ``SCUFRIS_MCP_SERVERS`` as JSON) - OFF by default.
    ``approve`` auto-approves the server's tools for unattended ``codex exec``;
    only set it for servers you trust, since the read-only sandbox is the only
    other guardrail. ``id`` must match ``^[A-Za-z0-9_]+$`` (a TOML key).
    """

    id: str
    command: str
    args: list[str] = Field(default_factory=list)
    approve: bool = True


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
    # Logging verbosity: DEBUG/INFO/WARNING/ERROR (env SCUFRIS_LOG_LEVEL). The CLI
    # `--debug`/`-v` flag overrides this to DEBUG.
    log_level: str = "INFO"
    # Built dashboard assets to serve at "/". Absent until the frontend is built.
    web_dist: Path = _REPO_ROOT / "web" / "dist"
    # Seconds the frontend waits between /api/stats polls (served to the client).
    poll_seconds: float = 2.0

    # --- Agent (Codex) ---------------------------------------------------
    # Off by default: the agent shells out to the `codex` CLI and needs a
    # `codex login`, so a fresh checkout runs the dashboard without it. See
    # tasks/20260719-153040/SPIKE.md.
    agent_enabled: bool = False
    # Model the agent drives (target GPT-5.5; a GPT-5.6 tier if the plan exposes
    # it). Empty string lets Codex pick its configured default.
    agent_model: str = "gpt-5.5"
    # "chatgpt" = Sign in with ChatGPT subscription (primary); "api_key" =
    # metered API key. Only affects `scufris login`; `codex` holds the auth.
    agent_auth_mode: Literal["chatgpt", "api_key"] = "chatgpt"
    # API key for the api_key auth mode (SCUFRIS_OPENAI_API_KEY).
    openai_api_key: str | None = None
    # Path to the `codex` binary; defaults to whatever is on PATH.
    codex_bin: str | None = None
    # Optional CODEX_HOME override where Codex stores its auth/session state.
    codex_home: Path | None = None
    # Seconds to wait for a `codex exec` turn before giving up.
    agent_timeout_seconds: float = 120.0
    # Expose the Scufris MCP tools (host_stats, tatr_*) to the agent. When on,
    # the agent registers the MCP server per codex-exec invocation via -c.
    agent_tools_enabled: bool = True
    # Extra MCP servers to register alongside the built-in Scufris one, declared
    # as JSON in SCUFRIS_MCP_SERVERS (empty by default - external servers are
    # opt-in; the operator supplies each binary and accepts its trust trade-off).
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)
