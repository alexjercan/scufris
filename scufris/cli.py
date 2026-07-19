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

from .agent import AgentUnavailable, build_agent, login
from .app import run_server
from .config import Settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scufris", description="Scuffed Jarvis")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="run the dashboard server (default)")
    sub.add_parser("login", help="authenticate the agent")
    chat = sub.add_parser("chat", help="run one agent turn and print the reply")
    chat.add_argument("prompt", help="the message to send to the agent")
    sub.add_parser("mcp-server", help="run the MCP tool server over stdio")
    return parser


async def _chat_once(settings: Settings, prompt: str) -> None:
    agent = build_agent(settings)
    try:
        reply = await agent.chat(prompt)
        print(reply.text)
    finally:
        await agent.aclose()


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    settings = Settings()

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
