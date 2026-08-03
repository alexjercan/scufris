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
import logging
import sys

from .agent import AgentUnavailable, StreamDone, StreamError, login
from .app import create_app
from .auth import AuthConfigError, validate_auth_config
from .backends import get_backend
from .config import Settings
from .env_bridge import ensure_api_base
from .logsetup import configure_logging
from .version import __version__

logger = logging.getLogger(__name__)


def run_server(settings: Settings | None = None) -> None:
    """Launch the dashboard app with uvicorn."""
    import uvicorn

    settings = settings or Settings()
    ensure_api_base(settings)
    # Un-forced: the CLI has usually already configured (honouring --debug); a
    # direct run_server() call configures from the setting instead.
    configure_logging(settings.log_level)
    # Check the auth posture BEFORE announcing a start: create_app would raise
    # anyway, but only after the log line has claimed the server is coming up.
    validate_auth_config(settings)
    logger.info(
        "starting scufris on %s:%d (agent %s)",
        settings.host,
        settings.port,
        "on" if settings.agent_enabled else "off",
    )
    # log_config=None: keep OUR logging config instead of uvicorn installing its
    # own, so scufris + uvicorn logs share one format/level.
    uvicorn.run(
        create_app(settings=settings),
        host=settings.host,
        port=settings.port,
        log_config=None,
        log_level=settings.log_level.lower(),
    )


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
    # Only the long form: `-v` is already taken by --debug above. The release
    # pipeline smoke-tests a built wheel with `scufris --version` before
    # publishing it, so this flag must not need a working config, a network,
    # or an agent backend - argparse prints and exits before any of that.
    parser.add_argument(
        "--version",
        action="version",
        version=f"scufris {__version__}",
        help="print the installed version and exit",
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
    sub.add_parser(
        "hash-password",
        parents=[common],
        help="hash a dashboard password for SCUFRIS_AUTH_PASSWORD_HASH",
    )
    return parser


def _hash_password_command() -> None:
    """Prompt for a password (no echo) and print its hash.

    The password itself never reaches a file, a log, or this process's argv -
    which is why this is a prompt rather than an argument. The printed hash is
    what goes in the sops dotenv as SCUFRIS_AUTH_PASSWORD_HASH.
    """
    import getpass

    from .auth import hash_password

    password = getpass.getpass("dashboard password: ")
    if not password:
        raise SystemExit("no password given")
    if password != getpass.getpass("repeat: "):
        raise SystemExit("passwords do not match")
    # To stdout, alone, so it can be piped; the guidance goes to stderr.
    print(f"SCUFRIS_AUTH_PASSWORD_HASH={hash_password(password)}")
    print(
        "\nAdd that line to your secrets file (`sops secrets/scufris.env` in "
        "nix.dotfiles), then rebuild. The password is not stored anywhere.",
        file=sys.stderr,
    )


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
    #
    # This path runs OUTSIDE the supervisor, so it must supply its own no-output
    # backstop: a stream that keeps producing events runs to completion, but a
    # genuinely stalled turn is bounded by `agent_heartbeat_seconds`. Without it,
    # a backend whose turn timeout is idle-unbounded (opencode's `read=None`, with
    # no internal idle guard) could hang the CLI forever. This mirrors the
    # supervisor's per-event heartbeat (supervisor._drain).
    agen = backend.stream(
        settings,
        prompt,
        is_orchestrator=True,
        permission_mode=settings.agent_permission_mode.value,
    )
    anext = agen.__anext__
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    anext(), timeout=settings.agent_heartbeat_seconds
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as exc:
                raise AgentUnavailable(
                    f"agent produced no output for {settings.agent_heartbeat_seconds}s"
                ) from exc
            if isinstance(event, StreamDone):
                reply_text = event.reply.text
            elif isinstance(event, StreamError):
                raise AgentUnavailable(event.detail)
    finally:
        aclose = getattr(agen, "aclose", None)
        if aclose is not None:
            await aclose()
    print(reply_text)


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(raw)
    settings = Settings()
    # The CLI owns the effective level: --debug beats the SCUFRIS_LOG_LEVEL setting.
    level = "DEBUG" if _wants_debug(raw) else settings.log_level
    configure_logging(level, force=True)

    if args.command in (None, "serve"):
        # A fail-closed auth config is an operator mistake, not a crash: report it
        # as one line rather than a traceback (see auth.validate_auth_config).
        try:
            run_server(settings)
        except AuthConfigError as exc:
            raise SystemExit(f"refusing to start: {exc}") from exc
        return

    if args.command == "mcp-server":
        from .mcp_server import main as mcp_main

        mcp_main()
        return

    if args.command == "hash-password":
        _hash_password_command()
        return

    try:
        if args.command == "login":
            asyncio.run(login(settings))
        elif args.command == "chat":
            asyncio.run(_chat_once(settings, args.prompt))
    except AgentUnavailable as exc:
        raise SystemExit(f"agent unavailable: {exc}") from exc
