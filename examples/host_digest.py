#!/usr/bin/env python
"""Print the digest in each state, so its wording can be judged before it ships.

    python examples/host_digest.py

The manual acceptance for this feature is "after a week of daily digests, the
operator still reads them", and that is a question about TEXT. So this renders the
five cases that decide whether it gets muted:

    1. the boring day        - `watch` says nothing at all
    2. the boring day        - `daily` says one line
    3. something is wrong    - the lead, the detail, what changed
    4. something recovered   - worth a message of its own
    5. a check broke         - named, with the rest of the digest intact

Nothing here touches the machine or the network: the check results are constructed
directly, because what is under inspection is the renderer's judgement about what to
say and in what order. `examples/host_inspect.py` is the one that reads the real
host.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Run from a checkout without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scufris.checks import (  # noqa: E402
    CheckResult,
    CheckRun,
    CheckState,
    escalation_for,
)
from scufris.digest import render_digest  # noqa: E402
from scufris_hostd import ActionKind  # noqa: E402

NOW = time.time()


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def show(label: str, digest: object) -> None:
    if digest is None:
        print(f"  {label}: (nothing is sent)")
        return
    text = getattr(digest, "text", "")
    verdict = getattr(digest, "verdict", "")
    print(f"  {label} [{verdict}]:")
    for line in str(text).splitlines():
        print(f"  | {line}")


def result(
    name: str, state: CheckState, headline: str, detail: list[str] | None = None
) -> CheckResult:
    return CheckResult(name=name, state=state, headline=headline, detail=detail or [])


def healthy() -> CheckRun:
    return CheckRun(
        at=NOW,
        results=[
            result("disk", CheckState.OK, "disks are fine (fullest: / at 62%)"),
            result("units", CheckState.OK, "nothing is in a failed state"),
            result("thermal", CheckState.OK, "temperatures are fine (hottest: 48C)"),
            result("store", CheckState.OK, "the store has 1204 unreachable path(s)"),
            result("flake", CheckState.OK, "the oldest pinned input is 9 days old"),
            result("scufris", CheckState.OK, "scufris 0.1.0 is healthy"),
        ],
    )


def troubled() -> CheckRun:
    run = healthy()
    run.results[0] = result(
        "disk",
        CheckState.CRIT,
        "/ is 96% full",
        [
            "/: 96% (452.1/470.0 GB)",
            "/home: 71% (330.0/470.0 GB)",
            "/boot: 34% (0.3/1.0 GB)",
        ],
    )
    run.results[1] = result(
        "units",
        CheckState.CRIT,
        "1 unit(s) failed: llama-server.service (system)",
        ["llama-server.service (system)"],
    )
    store = result(
        "store",
        CheckState.WARN,
        "the store holds 41203 unreachable path(s) and its filesystem is 96% full",
        ["collecting them frees space and touches no system generation"],
    )
    store.escalation = escalation_for(ActionKind.GC_STORE, {}, because=store.headline)
    run.results[3] = store
    return run


def broken() -> CheckRun:
    run = healthy()
    run.results[2] = result(
        "thermal",
        CheckState.FAILED,
        "the thermal check failed: OSError: [Errno 13] Permission denied",
    )
    run.results[4] = result(
        "flake", CheckState.FAILED, "the flake check timed out after 45s"
    )
    return run


def main() -> int:
    banner("1 + 2. a boring day")
    show("watch ", render_digest(healthy(), schedule="watch"))
    show("daily ", render_digest(healthy(), schedule="daily", always=True))
    print(
        "\n  (the point of the pair: `watch` costs nothing on a good day, and the\n"
        "   daily line is what makes its silence mean something)"
    )

    banner("3. something is wrong")
    yesterday = {name: "ok" for name in ("disk", "units", "thermal", "store", "flake")}
    show("watch ", render_digest(troubled(), previous=yesterday, schedule="watch"))
    escalating = [r for r in troubled().results if r.escalation is not None]
    for check in escalating:
        print(
            f"\n  (the {check.name} check would ALSO propose "
            f"{check.escalation.kind if check.escalation else ''} - if "
            "check_escalate_gc is on. It enters the approval queue; nothing is\n"
            "   collected until you approve it.)"
        )

    banner("4. something recovered")
    was_bad = {"disk": "crit", "units": "crit"}
    show("watch ", render_digest(healthy(), previous=was_bad, schedule="watch"))

    banner("5. a check broke")
    show("daily ", render_digest(broken(), schedule="daily", always=True))
    print(
        "\n  (a broken check is NAMED. A digest that quietly got shorter would read\n"
        "   as good news, which is the failure this shape exists to prevent.)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
