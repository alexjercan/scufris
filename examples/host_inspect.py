#!/usr/bin/env python
"""Run every host inspection against the real host and print it.

The cheapest end-to-end proof that ``scufris_host`` works on a real NixOS box,
and a re-runnable probe rig: when a report looks wrong in chat or on the
dashboard, run this and see exactly what the parsers made of the machine.

    python examples/host_inspect.py            # everything except the slow walks
    python examples/host_inspect.py --slow     # add the store walk and du
    python examples/host_inspect.py --json     # the structured overview instead

Nothing here mutates anything: every command in ``scufris_host`` is read-only,
and the garbage collection probe passes ``--dry-run`` unconditionally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run from a checkout without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scufris_host import (  # noqa: E402
    HostInspector,
    Scope,
    render,  # noqa: E402
)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slow",
        action="store_true",
        help="also run the store walk and the directory scan (tens of seconds)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the structured overview instead"
    )
    parser.add_argument(
        "--root", default=str(Path.home()), help="root for the directory scan"
    )
    args = parser.parse_args()

    inspector = HostInspector()

    if args.json:
        print(inspector.overview().model_dump_json(indent=2))
        return 0

    section("FAILED UNITS (system)")
    print(render.render_units(inspector.failed_units(scope=Scope.SYSTEM)))

    section("FAILED UNITS (user)")
    print(render.render_units(inspector.failed_units(scope=Scope.USER)))

    section("ONE UNIT")
    print(render.render_unit_status(inspector.unit_status("sshd.service")))

    section("JOURNAL (system, errors, last day)")
    print(
        render.render_journal(
            inspector.journal(priority="err", since="1 day ago", lines=15)
        )
    )

    section("STORAGE")
    print(render.render_storage(inspector.storage()))

    section("NETWORK")
    print(render.render_network(inspector.network()))

    section("THERMAL AND POWER")
    print(render.render_thermal(inspector.thermal()))

    section("WHAT PROVIDES systemctl")
    print(render.render_provider(inspector.what_provides("systemctl")))

    section("SYSTEM PROFILE")
    print(render.render_profile(inspector.profile(limit=10)))

    section("CLOSURE DIFF (previous -> current generation)")
    generations = inspector.generations()
    numbers = [g.number for g in generations.generations]
    if len(numbers) >= 2:
        print(
            render.render_closure_diff(inspector.closure_diff(numbers[1], numbers[0]))
        )
    else:
        print("fewer than two generations exist, so there is nothing to diff")

    section("FLAKE INPUTS")
    print(render.render_flake_status(inspector.flake_status()))

    if args.slow:
        section("RECLAIMABLE SPACE (walks the whole store)")
        print(render.render_reclaimable(inspector.reclaimable_space()))

        section(f"LARGEST DIRECTORIES under {args.root}")
        print(
            render.render_largest_directories(
                inspector.largest_directories(args.root, depth=1, limit=10)
            )
        )
    else:
        print("\n(skipped the store walk and directory scan; pass --slow for them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
