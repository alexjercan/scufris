"""Scufris command-line entry point.

Subcommands:
  serve            run the dashboard server (default when no subcommand)
  login            authenticate the agent (device-code or API key)
  chat "<prompt>"  run one agent turn and print the reply

`login` and `chat` require the operator-installed Codex toolchain and the agent
enabled; they exist so the operator can exercise the real subscription path.
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

    try:
        if args.command == "login":
            asyncio.run(login(settings))
        elif args.command == "chat":
            asyncio.run(_chat_once(settings, args.prompt))
    except AgentUnavailable as exc:
        raise SystemExit(f"agent unavailable: {exc}") from exc
