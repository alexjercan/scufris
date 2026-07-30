"""Application settings.

Loaded from the environment (prefix ``SCUFRIS_``) and an optional ``.env`` file
via pydantic-settings. See ``.env.example`` for the knobs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .enums import AuthMode, AuthPolicy, Backend, PermissionMode

# Repository root, derived from this file's location: <root>/scufris/config.py.
# In an editable dev install this points at the checkout, so the built frontend
# at <root>/web/dist is found without extra configuration.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_base_dirs() -> list[Path]:
    """The `sesh` set: dirs scanned one level deep for candidate projects."""
    home = Path.home()
    return [
        home / "personal",
        home / "personal" / "_tests",
        home / "work",
        home / "third-party",
    ]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCUFRIS_",
        env_file=".env",
        extra="ignore",
        # The settings store mutates whitelisted fields in place at runtime;
        # validate_assignment makes each `setattr` type-check (and coerce), so a
        # bad override is rejected at write time rather than corrupting the live
        # config.
        validate_assignment=True,
    )

    host: str = "127.0.0.1"
    port: int = 8000
    # Where scufris persists mutable runtime state (config overrides/profiles).
    # Env base seeds first boot; the store layers persisted overrides on top.
    state_dir: Path = Path.home() / ".local" / "state" / "scufris"
    # Base directories scanned ONE level deep for candidate projects (the
    # `sesh`-style discovery, minus tmux); "create" also mkdirs a new project under
    # one of these. Defaults to the sesh set; override via SCUFRIS_PROJECT_BASE_DIRS
    # as a colon-separated list ("~/personal:~/work") or a JSON array.
    project_base_dirs: list[Path] = Field(default_factory=_default_base_dirs)
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

    # --- Host inspection --------------------------------------------------
    # Seconds a /api/host/overview snapshot is reused before it is re-collected.
    # The overview shells out (systemctl, nixos-rebuild), so it gets its own much
    # slower cadence than /api/stats and a server-side cache on top: N dashboards
    # polling must not mean N nixos-rebuild invocations. Served to the client as
    # the interval it should poll at.
    host_overview_seconds: float = 30.0
    # The directory holding THIS host's NixOS flake, read (never written) to
    # report how old its pinned inputs are. Expanded at use time, since pydantic
    # stores a "~" env value verbatim.
    host_config_repo: Path = Path.home() / "personal" / "nix.dotfiles"

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
    agent_backend: Backend = Backend.CODEX
    # Model the agent drives (target GPT-5.5; a GPT-5.6 tier if the plan exposes
    # it). Empty string lets Codex pick its configured default. This is the CODEX
    # default model; claude agents use `claude_model` (below).
    agent_model: str = "gpt-5.5"
    # Default model for CLAUDE-backed agents. Kept separate so a claude agent
    # never shows a codex model like "gpt-5.5"; override via SCUFRIS_CLAUDE_MODEL.
    claude_model: str = "claude-opus-4-8"
    # The landing orchestrator's write posture (manual|edit|auto). Default AUTO -
    # a deliberate posture change: the orchestrator must do write work unattended
    # (run Bash tatr, create projects/agents), so it gets the full sandbox by
    # default; drop it to manual/edit here or from its settings page if that is
    # too permissive. Project agents are unaffected - they carry their own
    # permission_mode on the record (still defaulting manual); the orchestrator
    # has no agents.json row, so its mode is a setting, editable at runtime from
    # its settings page. Env SCUFRIS_AGENT_PERMISSION_MODE.
    agent_permission_mode: PermissionMode = PermissionMode.AUTO
    # Expose the `mock` backend (an in-process fake for dev/tests). Off in
    # production - agents can only be CREATED with the mock backend when this is
    # on; the resolver still resolves an already-persisted mock agent.
    enable_mock_backend: bool = False
    # The CODEX auth mode: "chatgpt" = Sign in with ChatGPT subscription
    # (primary); "api_key" = metered API key. Only affects `scufris login`;
    # `codex` holds the real auth. (For claude, see agent_claude_auth_mode.)
    agent_auth_mode: AuthMode = AuthMode.CHATGPT
    # The CLAUDE auth mode, reported per-agent for a claude-backed agent:
    # "claude_ai" = Sign in with claude.ai subscription (primary); "api_key" =
    # metered API key. Reported/effective only - the `claude` CLI holds the real
    # auth (there is no scufris claude-login flow yet).
    agent_claude_auth_mode: AuthMode = AuthMode.CLAUDE_AI
    # The OPENCODE auth mode. Always "local": opencode drives a self-hosted
    # llama.cpp server through a running `opencode serve` daemon, which needs no
    # subscription login (the daemon's HTTP Basic password, if any, is
    # SCUFRIS_OPENCODE_PASSWORD below, not an auth *mode*).
    agent_opencode_auth_mode: AuthMode = AuthMode.LOCAL
    # Base URL of the running `opencode serve` daemon the opencode backend drives.
    opencode_url: str = "http://127.0.0.1:4096"
    # HTTP Basic password for the daemon (username empty), if it was started with
    # OPENCODE_SERVER_PASSWORD. Unset for a loopback dev daemon.
    opencode_password: str | None = None
    # Default model for OPENCODE-backed agents, as the daemon's provider model id.
    # The provider (below) is prefixed on the wire: providerID/modelID.
    opencode_model: str = "gemma-4-26B-A4B-it"
    # The opencode provider id that routes to the llama.cpp server (declared in
    # the daemon's opencode.json; see examples/opencode/opencode.json).
    opencode_provider: str = "llamacpp"
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
    # No-output IDLE guard for a `codex app-server` turn: the max silence allowed
    # between app-server lines. It bounds each read, NOT the turn's total
    # wall-clock - a turn that keeps streaming (a long conversation turn, or a
    # spawned sub-agent working for minutes) runs to completion however long it
    # takes; only a genuinely hung app-server (no line for this long) is cut with
    # a timeout StreamError. This realizes the ADR-001 model (supervisor.py),
    # which dropped the old per-turn request timeout; the supervisor's coarser
    # `agent_heartbeat_seconds` is the sibling no-event stall guard one layer up.
    # The opencode backend reuses this same value as its httpx client timeout
    # (backends.py); aligning that reader to the idle semantics is task-2
    # (20260724-081804).
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
    # Auto-wake: when a sub-agent finishes a run awaiting a decision (a WAITING
    # outcome from request_input), having reported its result (a REPORTED outcome
    # from report_back) or having errored, GRANT the orchestrator a turn with the
    # message injected, so a stalled or finished loop self-heals without the operator
    # driving it. OFF by default: a wake runs the orchestrator (which defaults to
    # `auto` permission mode) unattended; the safe default is to poll
    # (pending_agents) instead. Bidirectional comms BC4 (spike 20260723-001256).
    auto_wake: bool = False
    # Expose the Scufris MCP tools (host_stats, list_agents, the control tools) to
    # the orchestrator. When on, the agent registers the MCP server per codex
    # invocation via -c (orchestrator-only; see agent._mcp_overrides).
    agent_tools_enabled: bool = True
    # Individual built-in Scufris tools to hide from the agent (by name, e.g.
    # ["disk_usage"]). Passed to the MCP server subprocess, which drops them at
    # startup, so a disabled tool genuinely cannot be called - not just hidden
    # in the UI. Editable at runtime from the settings page.
    disabled_tools: list[str] = Field(default_factory=list)
    # Path to the-den journal directory the `journal_*` MCP tools read/write via
    # the `today` CLI (SCUFRIS_DEN_PATH). None by default: with no den configured
    # the journal tools stay inert (they report a clear "not configured" message
    # and never shell out), so scufris runs safely on a box without the-den. Only
    # the orchestrator-only `den` MCP server carries it (agent.scufris_mcp_servers),
    # so a project sub-agent never reaches the operator's journal. Set it to the den
    # root (e.g. ~/personal/the-den) to enable; not runtime-mutable (env/.env only).
    den_path: Path | None = None

    # --- Dashboard authentication ----------------------------------------
    # Whether the HTTP surface requires an authenticated operator session
    # (SCUFRIS_AUTH_MODE). "auto" (default) resolves from the bind address: open
    # on loopback, mandatory on anything network-reachable - so `pytest`, the
    # examples and the mock backend need no login, while the deployed
    # 0.0.0.0 bind is protected without opting in. "required" forces it on even
    # for loopback; "disabled" turns it off and is REFUSED on a non-loopback bind.
    # Not runtime-mutable: a security posture must not be changeable through the
    # surface it protects. See tasks/20260729-125015/DECISION.md.
    auth_mode: AuthPolicy = AuthPolicy.AUTO
    # The operator's password hash (SCUFRIS_AUTH_PASSWORD_HASH), NOT the password.
    # Encoded `scrypt$n$r$p$salt$hash`; generate it with `scufris hash-password`.
    # Delivered like every other secret here - a line in the sops dotenv
    # (secrets/scufris.env) that reaches the unit as an EnvironmentFile. When
    # authentication is required and this is unset, the app refuses to start.
    auth_password_hash: str | None = None
    # The per-process MACHINE credential for the app's own MCP tool subprocesses,
    # minted by `create_app` at startup - NOT operator configuration, and never
    # persisted (it is absent from settings_store.WRITABLE_KEYS, so it is never
    # written to settings.json). It lives here rather than in `os.environ` so it
    # reaches ONLY the MCP servers that call the API: an env var would be
    # inherited by the agent CLI and therefore by every shell command the model
    # runs. Empty outside a running server. See tasks/20260729-125015/REVIEW.md
    # finding 2.
    auth_api_token: str = ""
    # Idle timeout: a session unused for this long stops working (default 12h, so
    # a phone left on the dashboard overnight asks for the password again).
    auth_session_idle_seconds: float = 12 * 60 * 60
    # Absolute cap: a session dies this long after LOGIN however actively it is
    # used (default 7 days). Sliding renewal must not mean immortal.
    auth_session_max_seconds: float = 7 * 24 * 60 * 60
    # Failed logins allowed from one source within the window before it is locked
    # out (SCUFRIS_AUTH_LOGIN_MAX_FAILURES / _WINDOW_SECONDS).
    auth_login_max_failures: int = 10
    auth_login_window_seconds: float = 15 * 60

    # --- Privileged host actions (scufris-hostd) -------------------------
    # The helper is a SEPARATE root system unit declared in nix.dotfiles
    # (services.scufris-hostd); the app only talks to its socket. With no
    # secret configured there is no privileged surface at all, and every
    # mutating host endpoint answers "not configured" rather than half-working.
    hostd_socket: Path = Path("/run/scufris-hostd/hostd.sock")
    # The shared secret every socket frame carries (SCUFRIS_HOSTD_SECRET, from
    # the same sops dotenv as the password hash).
    #
    # Unlike auth_api_token, this one is DELIVERED through the environment - the
    # unit's EnvironmentFile is how a sops secret reaches the process - so it is
    # in os.environ by construction and cannot merely be kept out of it. It must
    # instead be STRIPPED from every subprocess environment the model can reach
    # (SECRET_ENV_VARS below), because the agent CLI inherits our environment and
    # the socket is reachable by anything running as this user. Leaving it in
    # would hand the socket to precisely the caller this secret exists to keep
    # off it. See tasks/20260729-125029/DECISION.md and REVIEW.md findings R1.3
    # and R2.1.
    hostd_secret: str = ""
    # How often the approval queue listing reconciles with the helper (which owns
    # every proposal). The listing is polled by every open dashboard tab, so this
    # bounds "one socket round trip per tab per poll" without letting the queue go
    # stale for longer than a decision takes to matter. 0 reconciles every time.
    host_queue_refresh_seconds: float = 3.0

    # --- NixOS configuration changes (R3) --------------------------------
    # Which nixosConfiguration of the config repo THIS machine is. Empty means
    # "the hostname", which is what the operator's flake names its hosts by; a
    # wrong value is refused with the list of attributes that do exist rather
    # than a nix evaluation error.
    host_config_attr: str = ""
    # Wall clock for one `nix build` of the whole system. Two hours: a kernel or
    # nvidia rebuild is genuinely that slow, and an unbounded build would be a run
    # nobody can account for. The build is unprivileged and cancellable.
    host_config_build_timeout: float = 7200.0

    # --- Telegram frontend -----------------------------------------------
    # Bot API token for the Telegram frontend (SCUFRIS_TELEGRAM_BOT_TOKEN). When
    # set, the app launches an in-process long-poll bot that drives the SAME
    # orchestrator turn path as the landing chat; when unset (default) the bot
    # never starts. Get one from @BotFather.
    telegram_bot_token: str | None = None
    # Chat ids allowed to drive the bot (SCUFRIS_TELEGRAM_ALLOWED_CHAT_IDS). The
    # bot silently ignores updates from any other chat - this allowlist IS the
    # auth (no public webhook, no other guard). Accepts a comma/colon-separated
    # env string ("123,456") or a JSON array. Empty by default, which ignores
    # EVERYONE, so the bot is inert until you list your own chat id. NoDecode
    # hands the raw env string to `_split_chat_ids` instead of letting the
    # settings source JSON-decode it first (which would reject "123,456").
    telegram_allowed_chat_ids: Annotated[list[int], NoDecode] = Field(
        default_factory=list
    )
    # Stream the orchestrator turn live into the chat (SCUFRIS_TELEGRAM_STREAM).
    # When True (default) the bot renders the turn message-per-phase: a "thinking"
    # message that is edited as the reasoning streams, one widget message per tool
    # call, then the final answer as its own message. When False it falls back to
    # the calmer one-final-message-per-turn behaviour (reasoning/tool events are
    # not rendered; only the final answer is sent).
    telegram_stream: bool = True

    @field_validator("telegram_allowed_chat_ids", mode="before")
    @classmethod
    def _split_chat_ids(cls, value: object) -> object:
        # Accept a comma/colon-separated env string ("123,456" or "123:456") as
        # well as a JSON array; with NoDecode the source no longer parses either,
        # so this validator owns both forms. pydantic coerces each element to int
        # and rejects a non-numeric one.
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                return json.loads(text)
            return [seg for seg in re.split(r"[,:]", text) if seg.strip()]
        return value

    @field_validator("project_base_dirs", mode="before")
    @classmethod
    def _split_base_dirs(cls, value: object) -> object:
        # Accept a colon-separated env string ("~/personal:~/work") in addition to
        # a JSON array, since a PATH-style list is the natural way to set dirs.
        if isinstance(value, str):
            text = value.strip()
            if not text or text.startswith("["):
                return value  # empty or JSON -> let pydantic parse it
            return [seg for seg in text.split(":") if seg]
        return value

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
# The user-facing backends are "codex", "claude" and "opencode"; "mock" is a dev
# backend behind `enable_mock_backend`. Legacy persisted records may hold codex
# MODES ("app_server"/"exec") - those canonicalize to "codex". These helpers live
# in config (no heavy imports) so `agent_store` can validate/normalize without
# pulling in the backend runners.

_CANONICAL_BACKEND: dict[str, str] = {
    "codex": "codex",
    "app_server": "codex",
    "exec": "codex",
    "claude": "claude",
    "opencode": "opencode",
    "mock": "mock",
}


def canonical_backend(name: str) -> str:
    """Fold a backend id (incl. legacy codex modes) to its canonical name."""
    key = name.strip().lower()
    return _CANONICAL_BACKEND.get(key, key)


def available_backends(settings: "Settings") -> list[str]:
    """The backends an agent may be CREATED with, given the mock dev flag."""
    return ["codex", "claude", "opencode"] + (
        ["mock"] if settings.enable_mock_backend else []
    )


def auth_mode_for_backend(settings: "Settings", backend: str) -> AuthMode | None:
    """The reported auth mode for an agent on ``backend``: the codex mode for a
    codex agent (ChatGPT / API key), the claude mode for a claude agent (claude.ai
    / API key), and None for a backend with no login (mock). The subscription
    login differs per backend, so a claude agent must not report the codex mode."""
    canonical = canonical_backend(backend)
    if canonical == "codex":
        return settings.agent_auth_mode
    if canonical == "claude":
        return settings.agent_claude_auth_mode
    if canonical == "opencode":
        return settings.agent_opencode_auth_mode
    return None


# Friendly display labels for the backend ids (the server is the source of
# truth for the picker; the frontend mirrors these).
_BACKEND_LABELS: dict[str, str] = {
    "codex": "Codex",
    "claude": "Claude",
    "opencode": "Opencode",
    "mock": "Mock",
}


def backend_label(backend: str) -> str:
    """The human label for a backend id, falling back to the raw id."""
    return _BACKEND_LABELS.get(canonical_backend(backend), backend)


def default_model_for(settings: "Settings", backend: str) -> str:
    """The default model to stamp for a new agent on ``backend``."""
    canonical = canonical_backend(backend)
    if canonical == "claude":
        return settings.claude_model
    if canonical == "opencode":
        return settings.opencode_model
    return settings.agent_model


# The selectable model catalog per backend, offered as autocomplete suggestions
# in the create/settings model field. Not exhaustive/enforced - the field keeps a
# free-text escape so an operator can enter any model id the CLI accepts.
_BACKEND_MODELS: dict[str, list[str]] = {
    "codex": ["gpt-5.5", "gpt-5.6"],
    "claude": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
    "opencode": ["gemma-4-26B-A4B-it", "gemma-4-12B-it", "Qwen3.6-35B-A3B"],
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
# backends.py (codex --sandbox / claude --permission-mode). Default is manual.
# The option set lives on the PermissionMode enum; this tuple is kept for the
# membership check (values equal the enum members).
PERMISSION_MODES: tuple[str, ...] = tuple(m.value for m in PermissionMode)


def normalize_permission_mode(mode: str) -> PermissionMode:
    """Fold an input to a valid permission mode; unknown -> the safe default."""
    key = mode.strip().lower()
    return PermissionMode(key) if key in PERMISSION_MODES else PermissionMode.MANUAL


# Environment variables that must never reach a subprocess the model can drive.
#
# The agent CLI inherits this process's environment, and everything the model
# runs inherits the CLI's - so any credential sitting in os.environ is a
# credential the model holds. `agent.agent_subprocess_env` strips every name here.
#
# Two different situations, and the second is why this is a set rather than one
# `pop`:
#
# - SCUFRIS_API_TOKEN is minted in-process and deliberately never put in the
#   environment (20260729-125015 review round 1, finding 2), so stripping it
#   only guards against an operator or a stale shell having set it.
# - SCUFRIS_HOSTD_SECRET arrives THROUGH the environment - an EnvironmentFile is
#   how a sops secret reaches the unit - so it is present by construction and
#   cannot merely be "kept out of os.environ". It is the credential for the root
#   helper's socket, which is reachable by anything running as this user, the
#   agent CLI included. Without this strip a prompt-injected model applies host
#   actions directly and the operator approval is decoration (20260729-125029
#   review round 1, finding R1.3).
#
# The rest are secrets delivered the same way and equally useless to the agent
# CLI: the operator's password hash (offline-crackable), the Telegram bot token,
# and the provider credentials. None is needed by a turn - `scufris login` reads
# openai_api_key from Settings, not from the child environment.
#
# `test_every_secret_setting_is_stripped_from_the_agent_environment` asserts
# that every secret-shaped Settings field appears here, so a credential added
# later is covered by that test failing rather than by someone remembering.
SECRET_ENV_VARS: frozenset[str] = frozenset(
    {
        "SCUFRIS_API_TOKEN",
        "SCUFRIS_AUTH_API_TOKEN",
        "SCUFRIS_AUTH_PASSWORD_HASH",
        "SCUFRIS_HOSTD_SECRET",
        "SCUFRIS_OPENAI_API_KEY",
        "SCUFRIS_OPENCODE_PASSWORD",
        "SCUFRIS_TELEGRAM_BOT_TOKEN",
    }
)

# Field names that hold a credential, matched against Settings.model_fields by
# the test above. Keeping the PATTERN here rather than in the test means a field
# named `..._secret` cannot be added without either being stripped or the
# pattern being changed deliberately.
SECRET_FIELD_PATTERN = re.compile(
    r"secret|token|password|hash|api_key|credential", re.IGNORECASE
)
