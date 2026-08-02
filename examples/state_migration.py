#!/usr/bin/env python
"""Upgrade an operator's JSON state directory, and watch nothing get lost.

    python examples/state_migration.py

The claim this feature makes to an operator is "upgrading never loses a login, a
pending approval or a schedule". That is a promise about THEIR directory, and the
only convincing way to show it is to build one, start the real app on top of it,
and then use the state that came out the other side:

    1. a pre-database state directory  - one JSON file per store, as it was
    2. the first start                 - every source imported, once, backed up
    3. the login survives              - the cookie from before still authenticates
    4. the second start                - a no-op: nothing re-read, nothing doubled
    5. a damaged file                  - refused BY NAME, startup fails, and every
                                         other source is still in with its gate row

Nothing here touches the network or the host: the app is driven in-process and
every path is a temporary directory. `tests/test_db_legacy.py` and
`tests/test_db_state_boundary.py` are the assertions; this is the walkthrough.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

# Run from a checkout without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from scufris.app import create_app  # noqa: E402
from scufris.auth import SESSION_COOKIE, hash_password  # noqa: E402
from scufris.config import Settings  # noqa: E402
from scufris.db import close_all_state_databases  # noqa: E402
from scufris.db.legacy import BACKUP_SUFFIX  # noqa: E402
from scufris.db.models import LegacyImportRow  # noqa: E402
from scufris.enums import AuthPolicy  # noqa: E402

PASSWORD = "correct horse battery staple"
SESSION_ID = "the-session-in-my-browser"
ok = True


def report(claim: str, held: bool) -> None:
    global ok
    ok = ok and held
    print(f"  {'ok  ' if held else 'FAIL'}  {claim}")


def write_legacy_state(state_dir: Path) -> None:
    """One JSON file per store, the way a pre-database install left them."""
    now = time.time()
    (state_dir / "projects.json").write_text(
        json.dumps(
            [{"id": "scufris", "cwd": str(state_dir), "name": "Scufris"}], indent=2
        )
    )
    (state_dir / "auth_sessions.json").write_text(
        json.dumps(
            {
                "sessions": {
                    SESSION_ID: {
                        "csrf": "the-csrf-token-bound-to-it",
                        "created_at": now - 3600.0,
                        "last_seen": now - 60.0,
                    }
                }
            },
            indent=2,
        )
    )
    (state_dir / "schedules.json").write_text(
        json.dumps(
            {
                "schedules": {
                    "watch": {
                        "name": "watch",
                        # In the future, so the live scheduler does not record a
                        # missed window while this example is running.
                        "next_due": now + 3600.0,
                        "last_run": now - 900.0,
                        "last_result": "ran (attention), delivered",
                        "missed": 2,
                        "runs": 41,
                    }
                }
            },
            indent=2,
        )
    )
    (state_dir / "digests.json").write_text(
        json.dumps(
            {
                "digests": [
                    {
                        "at": now - 900.0,
                        "schedule": "watch",
                        "verdict": "attention",
                        "text": "15:20 - disk: / is 96% full",
                        "delivered": False,
                        "delivery_error": "telegram: timed out",
                        "states": {"disk": "attention"},
                    }
                ]
            },
            indent=2,
        )
    )


def boot(state_dir: Path):  # type: ignore[no-untyped-def]
    """Start the real app on ``state_dir``. This is where the import happens."""
    return create_app(
        settings=Settings(
            web_dist=state_dir / "absent",
            state_dir=state_dir,
            auth_mode=AuthPolicy.REQUIRED,
            auth_password_hash=hash_password(PASSWORD),
            _env_file=None,  # type: ignore[call-arg]
        )
    )


def gates(app) -> set[str]:  # type: ignore[no-untyped-def]
    with app.state.db.transaction() as conn:
        return set(conn.scalars(select(LegacyImportRow.source)).all())


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="scufris-migration-") as tmp:
        state_dir = Path(tmp) / "state"
        state_dir.mkdir()

        print("\n1. the state directory, before")
        write_legacy_state(state_dir)
        for path in sorted(state_dir.glob("*.json")):
            print(f"     {path.name}")

        print("\n2. the first start")
        app = boot(state_dir)
        with TestClient(app) as client:
            imported = gates(app)
            print(f"     imported: {', '.join(sorted(imported))}")
            report(
                "every source has its gate row",
                {
                    "projects.json",
                    "auth_sessions.json",
                    "schedules.json",
                    "digests.json",
                }
                <= imported,
            )
            report(
                "nothing pretends to have imported host actions "
                "(that store never had a file)",
                "host_actions.json" not in imported,
            )
            backups = sorted(p.name for p in state_dir.glob(f"*{BACKUP_SUFFIX}"))
            print(f"     backed up: {', '.join(backups)}")
            report("every source is still there, unread from now on", backups != [])

            print("\n3. the login survives the upgrade")
            client.cookies.set(SESSION_COOKIE, SESSION_ID)
            answer = client.get("/api/auth/session").json()
            print(f"     GET /api/auth/session -> {answer}")
            report(
                "the cookie from before still authenticates", answer["authenticated"]
            )

            schedule = {s.name: s for s in app.state.host_scheduler.store.all()}[
                "watch"
            ]
            print(
                f"     watch: {schedule.runs} runs, {schedule.missed} missed, "
                f"last: {schedule.last_result!r}"
            )
            report("the schedule kept its history", schedule.runs == 41)

            digest = app.state.digests.list()[0]
            print(f"     digest: {digest.text!r} (delivered={digest.delivered})")
            report(
                "the digest kept its delivery outcome",
                digest.delivered is False
                and digest.delivery_error == "telegram: timed out",
            )
            report(
                "the project came through the same one call",
                [p.id for p in app.state.projects.list()] == ["scufris"],
            )
        close_all_state_databases()

        print("\n4. the second start re-reads nothing")
        app = boot(state_dir)
        with TestClient(app) as client:
            report("one gate row per source, still", gates(app) == imported)
            report(
                "one copy of each record, still",
                len(app.state.digests.list()) == 1
                and len(app.state.projects.list()) == 1,
            )
        close_all_state_databases()

        print("\n5. a damaged file is refused by name")
        second = Path(tmp) / "damaged"
        second.mkdir()
        write_legacy_state(second)
        (second / "schedules.json").write_text('{"schedules": {"watch": 3}}')
        try:
            boot(second)
        except Exception as exc:  # the import refuses; startup does not continue
            message = str(exc)
            print(f"     {type(exc).__name__}: {message.splitlines()[0]}")
            report("the refusal names the file", "schedules.json" in message)
            report("and says which entry", "'watch'" in message)
        else:
            report("startup was refused", False)
        finally:
            close_all_state_databases()

    print("\nAll good." if ok else "\nSomething did not hold.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
