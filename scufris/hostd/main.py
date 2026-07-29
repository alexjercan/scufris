"""The ``scufris-hostd`` entry point: the only privileged process scufris has.

Run as root by a NixOS system unit (``nix/scufris-hostd.nix``), never by the
app. It fails closed on every missing precondition - no secret file, an empty
secret, an unwritable audit log - because a helper that starts anyway would be a
root socket with no credential in front of it.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from ..logsetup import configure_logging
from .audit import DEFAULT_KEEP, DEFAULT_MAX_BYTES, AuditLog
from .engine import DEFAULT_TTL_SECONDS, HostdEngine
from .server import HostdServer

logger = logging.getLogger(__name__)

DEFAULT_SOCKET = Path("/run/scufris-hostd/hostd.sock")
DEFAULT_AUDIT_LOG = Path("/var/log/scufris-hostd/audit.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scufris-hostd",
        description=(
            "The privileged host-action helper. Speaks a closed set of typed "
            "verbs over a unix socket; there is no shell verb at any privilege."
        ),
    )
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument(
        "--secret-file",
        type=Path,
        required=True,
        help=(
            "File holding the shared secret callers must present. Deployed from "
            "the sops dotenv; the helper refuses to start without it."
        ),
    )
    parser.add_argument("--audit-log", type=Path, default=DEFAULT_AUDIT_LOG)
    parser.add_argument("--audit-max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--audit-keep", type=int, default=DEFAULT_KEEP)
    parser.add_argument(
        "--group",
        default=None,
        help="Group given read/write on the socket (the operator's group).",
    )
    parser.add_argument("--ttl-seconds", type=float, default=DEFAULT_TTL_SECONDS)
    parser.add_argument("--log-level", default="INFO")
    return parser


def read_secret(path: Path) -> str:
    """Read the shared secret, or explain exactly why the helper will not start."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(
            f"scufris-hostd: cannot read the secret file {path}: {exc}"
        ) from exc
    secret = raw.strip()
    if not secret:
        raise SystemExit(
            f"scufris-hostd: the secret file {path} is empty. Add "
            "SCUFRIS_HOSTD_SECRET to the sops dotenv before enabling the helper."
        )
    return secret


async def _serve(args: argparse.Namespace) -> None:
    secret = read_secret(args.secret_file)
    audit = AuditLog(
        args.audit_log,
        max_bytes=args.audit_max_bytes,
        keep=args.audit_keep,
        # The secret is redacted out of every record by value, so a message that
        # happens to quote a frame cannot turn the log into the leak.
        secrets=frozenset({secret}),
    )
    engine = HostdEngine(audit, ttl_seconds=args.ttl_seconds)
    server = HostdServer(
        engine, secret=secret, socket_path=args.socket, group=args.group
    )
    await server.start()
    try:
        await server.serve_forever()
    finally:
        await server.aclose()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        return 0
    except SystemExit:
        raise
    except ValueError as exc:
        # A refused precondition (no secret, unknown group) - not a crash.
        print(f"scufris-hostd: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
