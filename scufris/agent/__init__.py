"""The Scufris agent backend: the codex app-server runner and shared agent types.

This package holds the codex ``app-server`` streaming runner, the streaming event
types, and ``login`` - the low-level plumbing the swappable ``AgentBackend``
implementations in ``backends`` drive. The orchestrator and every agent run
through ``get_backend(...).stream()``, which streams token-by-token via the
app-server. The default codex path drives the ``codex`` CLI (nixpkgs `codex`,
"Sign in with ChatGPT" subscription) through a ``codex app-server`` subprocess.

We use the CLI rather than the ``openai-codex`` Python SDK because the SDK bundles
a prebuilt `codex` binary that does not build in the uv2nix venv (see
LESSONS.md `codex-binary-breaks-uv2nix-venv`); the nixpkgs `codex` runs fine
on NixOS and shares its auth under ``CODEX_HOME``. Using a ChatGPT subscription
programmatically is a personal-use gray area, so the agent is off unless the
operator enables it and has run ``codex login``.

The modules:

- ``events``    - the reply, the ``StreamEvent`` union, and the read ceiling.
- ``env``       - the codex binary, every child's environment, and ``login``.
- ``mcp``       - which scufris MCP servers a turn registers, and codex's
                  rendering of them.
- ``appserver`` - steering, the JSON-RPC handshake, and the turn's event stream.
"""

from .appserver import (
    _appserver_event,
    _git_writable_roots,
    _sandbox_overrides,
    _steer,
    _stream_app_server,
)
from .env import _codex_env, agent_subprocess_env, login
from .events import (
    STREAM_READ_LIMIT,
    AgentReply,
    AgentUnavailable,
    StreamDone,
    StreamError,
    StreamEvent,
    StreamReasoningDelta,
    StreamSessionStarted,
    StreamTextDelta,
    StreamTool,
    TokenUsage,
    ToolCall,
)
from .mcp import ScufrisMcpServer, _mcp_overrides, scufris_mcp_servers

__all__ = [
    "STREAM_READ_LIMIT",
    "AgentReply",
    "AgentUnavailable",
    "ScufrisMcpServer",
    "StreamDone",
    "StreamError",
    "StreamEvent",
    "StreamReasoningDelta",
    "StreamSessionStarted",
    "StreamTextDelta",
    "StreamTool",
    "TokenUsage",
    "ToolCall",
    "agent_subprocess_env",
    "login",
    "scufris_mcp_servers",
    # Internal to scufris, but part of this package's import surface: `backends`
    # drives the turn through `_stream_app_server`, and the tests reach the turn
    # builders directly. Listed so the re-export is deliberate rather than a
    # side effect of the facade's imports.
    "_appserver_event",
    "_codex_env",
    "_git_writable_roots",
    "_mcp_overrides",
    "_sandbox_overrides",
    "_steer",
    "_stream_app_server",
]
