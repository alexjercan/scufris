"""Codex session introspection.

Reads codex's on-disk rollout files (JSONL under ``$CODEX_HOME/sessions``) to
expose what the ``codex exec --json`` stream does not: the list of past sessions
(so the UI can switch between them), a per-session context snapshot (window +
token usage + turn/tool counts), and the account usage/quota (the weekly rate
limit). Everything here is read-only apart from ``delete_session``.

Codex already records all of this on disk, so this harvests what exists rather
than adding subprocess calls. The functions take an explicit ``codex_home`` and
``cwd`` so tests can drive them against a temp directory of fake rollout files,
with no codex binary in sight.

The modules:

- ``steering``   - the ``[scufris-tools]`` turn preambles and their inverse.
- ``models``     - the data models, importing nothing from scufris but config.
- ``rollout``    - finding and iterating rollouts; the session list and context.
- ``transcript`` - re-rendering a conversation, merging reasoning, forking.
- ``usage``      - account quota and the on-disk footprint.
"""

from .models import (
    RateWindow,
    SessionContext,
    SessionInfo,
    TokenUsage,
    ToolCall,
    TranscriptMessage,
    UsageQuota,
)
from .rollout import (
    delete_session,
    list_sessions,
    read_context,
    resolve_codex_home,
    rollout_mtime,
)
from .steering import (
    AGENT_STEERING_PREAMBLE,
    HOST_STEERING_PREAMBLE,
    STEERING_PREAMBLE,
    strip_steering,
)
from .transcript import (
    FORK_CONTEXT_TURNS,
    format_fork_seed,
    merge_reasoning,
    read_transcript,
    reasoning_fingerprint,
)
from .usage import MemoryFootprint, read_memory_footprint, read_usage

__all__ = [
    "AGENT_STEERING_PREAMBLE",
    "FORK_CONTEXT_TURNS",
    "HOST_STEERING_PREAMBLE",
    "STEERING_PREAMBLE",
    "MemoryFootprint",
    "RateWindow",
    "SessionContext",
    "SessionInfo",
    "TokenUsage",
    "ToolCall",
    "TranscriptMessage",
    "UsageQuota",
    "delete_session",
    "format_fork_seed",
    "list_sessions",
    "merge_reasoning",
    "read_context",
    "read_memory_footprint",
    "read_transcript",
    "read_usage",
    "reasoning_fingerprint",
    "resolve_codex_home",
    "rollout_mtime",
    "strip_steering",
]
