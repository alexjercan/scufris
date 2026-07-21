"""Application settings.

Loaded from the environment (prefix ``SCUFRIS_``) and an optional ``.env`` file
via pydantic-settings. See ``.env.example`` for the knobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# An MCP server id must be a bare TOML key (it becomes `mcp_servers.<id>` in a
# codex `-c` override); anything else would emit a malformed / injected config.
# Enforced when a server is ADDED via the settings endpoint (a user action gets
# a clear rejection); env-declared servers with a bad id are skipped by
# `_mcp_overrides` instead, so a stray env entry never crashes startup.
SERVER_ID_RE = r"^[A-Za-z0-9_]+$"


class McpServerSpec(BaseModel):
    """An extra MCP server to register with codex, beyond the built-in Scufris one.

    Declared in config (e.g. ``SCUFRIS_MCP_SERVERS`` as JSON) - OFF by default.
    ``approve`` auto-approves the server's tools for an unattended codex run;
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
        env_prefix="SCUFRIS_",
        env_file=".env",
        extra="ignore",
        # The settings store mutates whitelisted fields in place at runtime;
        # validate_assignment makes each `setattr` type-check (and coerce, e.g.
        # a list of dicts into McpServerSpec), so a bad override is rejected at
        # write time rather than corrupting the live config.
        validate_assignment=True,
    )

    host: str = "127.0.0.1"
    port: int = 8000
    # Where scufris persists mutable runtime state (config overrides/profiles).
    # Env base seeds first boot; the store layers persisted overrides on top.
    state_dir: Path = Path.home() / ".local" / "state" / "scufris"
    # When false, the settings store refuses every write (a read-only server);
    # the writable-config endpoints return 403. On by default (single-operator
    # local tool); safety is the mutable-key whitelist + a UI confirm.
    settings_writable: bool = True
    # Logging verbosity: DEBUG/INFO/WARNING/ERROR (env SCUFRIS_LOG_LEVEL). The CLI
    # `--debug`/`-v` flag overrides this to DEBUG.
    log_level: str = "INFO"
    # Built dashboard assets to serve at "/". Absent until the frontend is built.
    web_dist: Path = _REPO_ROOT / "web" / "dist"
    # Seconds the frontend waits between /api/stats polls (served to the client).
    poll_seconds: float = 2.0

    # --- Agent (Codex) ---------------------------------------------------
    # On by default. The agent shells out to the `codex` CLI and needs a
    # `codex login` (see tasks/20260719-153040/SPIKE.md); with no login the chat
    # endpoints return a clear "run codex login" error but the dashboard still
    # serves. To develop/test without codex at all, use `agent_backend=mock`.
    agent_enabled: bool = True
    # Which backend the landing orchestrator chat uses (the same vocabulary an
    # agent is created with):
    #   "codex" (default) - the `codex` CLI, streaming token-by-token + reasoning
    #       + live events over the app-server JSON-RPC protocol.
    #   "claude" - the `claude` Code headless backend.
    #   "mock" - an in-process fake that needs no login or network (dev/tests;
    #       selectable only when `enable_mock_backend` is on).
    # Legacy codex MODE ids ("app_server"/"exec", from before the codex/claude
    # unification) are coerced to "codex" below so old env/state still loads.
    agent_backend: Literal["codex", "claude", "mock"] = "codex"
    # Model the agent drives (target GPT-5.5; a GPT-5.6 tier if the plan exposes
    # it). Empty string lets Codex pick its configured default. This is the CODEX
    # default model; claude agents use `claude_model` (below).
    agent_model: str = "gpt-5.5"
    # Default model for CLAUDE-backed agents. Kept separate so a claude agent
    # never shows a codex model like "gpt-5.5"; override via SCUFRIS_CLAUDE_MODEL.
    claude_model: str = "claude-opus-4-8"
    # Expose the `mock` backend (an in-process fake for dev/tests). Off in
    # production - agents can only be CREATED with the mock backend when this is
    # on; the resolver still resolves an already-persisted mock agent.
    enable_mock_backend: bool = False
    # "chatgpt" = Sign in with ChatGPT subscription (primary); "api_key" =
    # metered API key. Only affects `scufris login`; `codex` holds the auth.
    agent_auth_mode: Literal["chatgpt", "api_key"] = "chatgpt"
    # API key for the api_key auth mode (SCUFRIS_OPENAI_API_KEY).
    openai_api_key: str | None = None
    # Path to the `codex` binary; defaults to whatever is on PATH.
    codex_bin: str | None = None
    # Optional CODEX_HOME override where Codex stores its auth/session state.
    codex_home: Path | None = None
    # Path to the `claude` binary (Claude Code headless backend); PATH by default.
    claude_bin: str | None = None
    # Where Claude Code stores its session transcripts (default ~/.claude); the
    # per-project session files live under <claude_home>/projects/<cwd-hash>/.
    claude_home: Path | None = None
    # Per-turn wall-clock deadline for a `codex app-server` turn: the runner
    # yields a timeout StreamError once it is exceeded. The supervisor's
    # `agent_heartbeat_seconds` is a separate no-output stall guard.
    agent_timeout_seconds: float = 120.0
    # Max agent runs the supervisor executes concurrently; further runs queue.
    # Turns of the same agent still serialize regardless of this cap. Startup
    # config (read once when the supervisor is built), not a live settings knob.
    agent_max_concurrent: int = 4
    # Stall guard for a supervised run: if it emits no event for this long it is
    # cancelled as hung. Generous so a legitimately slow turn (a multi-minute
    # codex/claude run, or a long tool call) is never killed for being slow -
    # this replaces the old request timeout, it does not reinstate it.
    agent_heartbeat_seconds: float = 600.0
    # Expose the Scufris MCP tools (host_stats, tatr_*) to the agent. When on,
    # the agent registers the MCP server per codex invocation via -c.
    agent_tools_enabled: bool = True
    # Individual built-in Scufris tools to hide from the agent (by name, e.g.
    # ["tatr_new"]). Passed to the MCP server subprocess, which drops them at
    # startup, so a disabled tool genuinely cannot be called - not just hidden
    # in the UI. Editable at runtime from the settings page.
    disabled_tools: list[str] = Field(default_factory=list)
    # Extra MCP servers to register alongside the built-in Scufris one, declared
    # as JSON in SCUFRIS_MCP_SERVERS (empty by default - external servers are
    # opt-in; the operator supplies each binary and accepts its trust trade-off).
    mcp_servers: list[McpServerSpec] = Field(default_factory=list)

    @field_validator("agent_backend", mode="before")
    @classmethod
    def _coerce_legacy_backend(cls, value: object) -> object:
        # Legacy codex MODE ids from before the codex/claude unification
        # ("app_server", the old default, and the retired "exec") both mean the
        # codex backend now, so old env/state keeps loading.
        if isinstance(value, str) and value.strip().lower() in {"app_server", "exec"}:
            return "codex"
        return value


# --- agent backend surface ---------------------------------------------------
#
# The two user-facing backends are "codex" and "claude"; "mock" is a dev backend
# behind `enable_mock_backend`. Legacy persisted records may hold codex MODES
# ("app_server"/"exec") - those canonicalize to "codex". These helpers live in
# config (no heavy imports) so `agent_store` can validate/normalize without
# pulling in the backend runners.

_CANONICAL_BACKEND: dict[str, str] = {
    "codex": "codex",
    "app_server": "codex",
    "exec": "codex",
    "claude": "claude",
    "mock": "mock",
}


def canonical_backend(name: str) -> str:
    """Fold a backend id (incl. legacy codex modes) to its canonical name."""
    key = name.strip().lower()
    return _CANONICAL_BACKEND.get(key, key)


def available_backends(settings: "Settings") -> list[str]:
    """The backends an agent may be CREATED with, given the mock dev flag."""
    return ["codex", "claude"] + (["mock"] if settings.enable_mock_backend else [])


# Friendly display labels for the backend ids (the server is the source of
# truth for the picker; the frontend mirrors these).
_BACKEND_LABELS: dict[str, str] = {
    "codex": "Codex",
    "claude": "Claude",
    "mock": "Mock",
}


def backend_label(backend: str) -> str:
    """The human label for a backend id, falling back to the raw id."""
    return _BACKEND_LABELS.get(canonical_backend(backend), backend)


def default_model_for(settings: "Settings", backend: str) -> str:
    """The default model to stamp for a new agent on ``backend``."""
    if canonical_backend(backend) == "claude":
        return settings.claude_model
    return settings.agent_model


# The selectable model catalog per backend, offered as autocomplete suggestions
# in the create/settings model field. Not exhaustive/enforced - the field keeps a
# free-text escape so an operator can enter any model id the CLI accepts.
_BACKEND_MODELS: dict[str, list[str]] = {
    "codex": ["gpt-5.5", "gpt-5.6"],
    "claude": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
    "mock": ["mock"],
}


def models_for(settings: "Settings", backend: str) -> list[str]:
    """The suggested models for ``backend``, with the configured default first
    (so the picker always offers the effective default even if it was overridden
    via env to something outside the built-in catalog)."""
    catalog = list(_BACKEND_MODELS.get(canonical_backend(backend), []))
    default = default_model_for(settings, backend)
    if default and default not in catalog:
        catalog.insert(0, default)
    return catalog


# An agent's write posture, Claude-style. Each maps to a per-backend flag in
# backends.py (codex --sandbox / claude --permission-mode). Default is manual:
#   manual = read-only (observe/plan), edit = may edit project files,
#   auto = edit + run commands unattended (full access).
PERMISSION_MODES: tuple[str, ...] = ("manual", "edit", "auto")


def normalize_permission_mode(mode: str) -> str:
    """Fold an input to a valid permission mode; unknown -> the safe default."""
    key = mode.strip().lower()
    return key if key in PERMISSION_MODES else "manual"
