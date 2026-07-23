"""Scufris command-line entry point.

Subcommands:
  serve            run the dashboard server (default when no subcommand)
  login            authenticate the agent (device-code or API key)
  chat "<prompt>"  run one agent turn and print the reply
  mcp-server       run the Scufris MCP tool server over stdio (spawned by Codex)

`login`, `chat` and `mcp-server` relate to the agent; the dashboard runs without
them.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .agent import AgentUnavailable, StreamDone, StreamError, login
from .app import run_server
from .backends import get_backend
from .config import Settings
from .logsetup import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    # A shared parent carries -v/--debug so argparse ACCEPTS it both before and
    # after the subcommand. argparse's own merge of a parent flag across
    # sub/parent namespaces is unreliable, so the effective value is read from
    # argv by `_wants_debug` rather than trusted from `args.debug`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v", "--debug", action="store_true", help="verbose (DEBUG) logging"
    )
    parser = argparse.ArgumentParser(
        prog="scufris", description="Scuffed Jarvis", parents=[common]
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", parents=[common], help="run the dashboard server")
    sub.add_parser("login", parents=[common], help="authenticate the agent")
    chat = sub.add_parser(
        "chat", parents=[common], help="run one agent turn and print the reply"
    )
    chat.add_argument("prompt", help="the message to send to the agent")
    sub.add_parser(
        "mcp-server", parents=[common], help="run the MCP tool server over stdio"
    )
    return parser


def _wants_debug(argv: list[str]) -> bool:
    """Whether -v/--debug appears anywhere in the args (position-independent)."""
    return "--debug" in argv or "-v" in argv


async def _chat_once(settings: Settings, prompt: str) -> None:
    """Run one fresh agent turn through the configured backend and print the reply.

    A one-shot CLI turn: no session is resumed. It IS an orchestrator turn, so it
    runs with the orchestrator's configured write posture
    (``settings.agent_permission_mode``) - one orchestrator, one posture, whether
    reached from the dashboard or the CLI.
    """
    if not settings.agent_enabled:
        raise AgentUnavailable(
            "agent is disabled. Set SCUFRIS_AGENT_ENABLED=1 and run `codex login` "
            "(or `scufris login`) to enable it."
        )
    backend = get_backend(settings.agent_backend)
    reply_text = ""
    # The one-shot CLI chat talks to the main agent (the orchestrator), so it
    # gets the orchestrator-only scufris tools and their steering.
    async for event in backend.stream(
        settings,
        prompt,
        is_orchestrator=True,
        permission_mode=settings.agent_permission_mode.value,
    ):
        if isinstance(event, StreamDone):
            reply_text = event.reply.text
        elif isinstance(event, StreamError):
            raise AgentUnavailable(event.detail)
    print(reply_text)


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(raw)
    settings = Settings()
    # The CLI owns the effective level: --debug beats the SCUFRIS_LOG_LEVEL setting.
    level = "DEBUG" if _wants_debug(raw) else settings.log_level
    configure_logging(level, force=True)

    if args.command in (None, "serve"):
        run_server(settings)
        return

    if args.command == "mcp-server":
        from .mcp_server import main as mcp_main

        mcp_main()
        return

    try:
        if args.command == "login":
            asyncio.run(login(settings))
        elif args.command == "chat":
            asyncio.run(_chat_once(settings, args.prompt))
    except AgentUnavailable as exc:
        raise SystemExit(f"agent unavailable: {exc}") from exc
