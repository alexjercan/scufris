"""R3: what the operator is shown before a configuration is switched to.

The preview here is deliberately NARROWER than the spike sketched, and the
reason is worth stating at the top of the file because it looks like a missing
feature.

**The unit-restart list is not shown, because obtaining it would mean running the
proposed configuration's own code as root before anyone approved it.**
``<toplevel>/bin/switch-to-configuration dry-activate`` is the only thing that
can produce that list, it requires root (measured: it refuses outright as the
operator), and the binary it runs comes FROM the toplevel being previewed - which
is a configuration an agent wrote and nobody has approved yet. Running it at
propose time would mean the framework's first promise ("proposing changes
nothing") depends on an unapproved store path choosing to be well behaved.
``dry_activate`` therefore has neither a verb nor a place in the preview.

What the preview does show comes from tools that read metadata rather than
execute the configuration:

- ``nix store diff-closures``, which is what actually answers "what changes",
- the generation and store path the system is on now,
- the revision the proposal was built from, so the operator can go read the diff
  in git - which is the real review surface for a config change.

Two measured traps this module exists to handle:

1. ``nix store diff-closures`` prints NOTHING and exits 0 when the two closures
   are identical, so "no change" and "the command failed" are byte-identical in
   its output. The exit status is checked first and "no closure change" is
   stated explicitly.
2. It emits ANSI colour codes and non-ASCII glyphs even when its output is a
   pipe (measured on this host: ``NO_COLOR=1`` does not suppress them). They are
   stripped here, at the source, rather than in each surface that renders a
   preview.
"""

from __future__ import annotations

import re

from scufris_host import (
    Availability,
    Generation,
    Outcome,
    Runner,
    list_generations,
    nix_cli,
)

from .actions import (
    GENERATION_TIMEOUT,
    SWITCH_UNIT,
    ActionKind,
    Plan,
    generation_link,
)
from .files import Files
from .preview import Fingerprint, Preview, PreviewKind, Reversal

# The symlink that says what is running right now.
CURRENT_SYSTEM = "/run/current-system"

# A closure diff walks two closures and reads every path's size; on this host a
# full nixpkgs bump answered in a few seconds.
DIFF_TIMEOUT = 300.0
IS_ACTIVE_TIMEOUT = 20.0

# How many closure-diff lines the preview carries. Measured: a nixpkgs bump
# produced 398 lines, which is a wall rather than a preview. The rest are
# counted, never silently dropped.
MAX_DIFF_LINES = 60

# ANSI colour, which `nix store diff-closures` emits even into a pipe.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# The two non-ASCII glyphs nix uses in a closure diff. This repo's surfaces are
# ASCII, and a mojibake arrow in an approval prompt is exactly the kind of
# detail that makes an operator stop trusting the text.
_GLYPHS = {"\u2192": "->", "\u2205": "(none)"}


def _clean(line: str) -> str:
    text = _ANSI.sub("", line)
    for glyph, replacement in _GLYPHS.items():
        text = text.replace(glyph, replacement)
    return text.rstrip()


def current_system(files: Files) -> str:
    """The store path the running system is, or empty when it cannot be read."""
    return files.resolve(CURRENT_SYSTEM)


def current_generation(runner: Runner) -> Generation | None:
    """The generation the system is on, or None when the list is unreadable."""
    listing = list_generations(runner, timeout=GENERATION_TIMEOUT)
    if not listing.ok:
        return None
    return listing.current


def switch_in_flight(runner: Runner) -> str:
    """Why an activation must not start right now, or empty when it may.

    A ``switch-to-configuration`` already running - started by this helper or by
    the operator's own ``nixos-rebuild`` - is the one state where beginning
    another one is actively destructive: two activations interleaving on one
    machine leave a system that matches neither configuration. Both run in the
    same transient unit name (``actions.SWITCH_UNIT``), so systemd would refuse
    the second anyway; this turns that into a sentence, and it runs BEFORE the
    system profile is touched rather than after.

    Being unable to ASK is also a refusal. If systemctl cannot be reached, this
    helper cannot know whether an activation is in flight, and "probably not" is
    not good enough for the one action that has no safe half-way point.
    """
    unit = f"{SWITCH_UNIT}.service"
    result = runner(["systemctl", "is-active", "--", unit], timeout=IS_ACTIVE_TIMEOUT)
    if result.outcome in (Outcome.MISSING, Outcome.DENIED, Outcome.TIMEOUT):
        return (
            "cannot tell whether another switch-to-configuration is already "
            f"running ({result.reason()}), and an activation must not start on a "
            "guess"
        )
    state = result.stdout.strip().splitlines()
    now = state[0].strip() if state else ""
    if now in ("active", "activating", "reloading", "deactivating"):
        return (
            f"{unit} is {now}: a switch-to-configuration is already running on "
            "this host, started either by this helper or by nixos-rebuild. Two "
            "activations at once leave a system that matches neither "
            "configuration, so this one is refused until that finishes"
        )
    return ""


def closure_diff(
    runner: Runner, before: str, after: str
) -> tuple[list[str], bool, str]:
    """``(lines, changed, caveat)`` for the closure difference between two systems.

    ``changed`` is the answer to the measured trap: empty output on a successful
    run means the closures are IDENTICAL, which is a real and reassuring answer,
    and it must never render the same as a preview that failed to run. ``caveat``
    is non-empty only when the diff could not be produced at all.
    """
    result = runner(
        nix_cli("store", "diff-closures", before, after), timeout=DIFF_TIMEOUT
    )
    if not result.ok:
        return [], False, result.reason()
    lines = [_clean(line) for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return [], False, ""
    shown = lines[:MAX_DIFF_LINES]
    if len(lines) > len(shown):
        shown.append(f"  ... and {len(lines) - len(shown)} more package changes")
    return shown, True, ""


def _no_execution_note() -> str:
    return (
        "the list of units that would restart is NOT shown: the only thing that "
        "can produce it is this configuration's own switch-to-configuration, run "
        "as root, and running an unapproved configuration's code to preview it "
        "would defeat the approval. Read the commit's diff for what changed in "
        "the configuration itself."
    )


def _command_lines(plan: Plan) -> list[str]:
    lines: list[str] = ["commands, in order:"]
    for index, step in enumerate(plan.steps, start=1):
        if step.label:
            lines.append(f"  {index}. {step.label}")
        lines.append(f"     $ {' '.join(step.argv)}")
    return lines


def _reversal_to(generation: Generation | None, *, leaving: str) -> Reversal:
    """Rolling back to the generation the system is on NOW."""
    if generation is None:
        return Reversal(
            possible=False,
            summary=(
                "the current generation number could not be read, so no rollback "
                "target can be recorded. Without it an undo would be a guess at "
                "which generation to return to"
            ),
        )
    return Reversal(
        possible=True,
        summary=(
            f"roll back to generation {generation.number}"
            + (f" ({generation.date})" if generation.date else "")
            + f", which is what {leaving} right now. A rollback is itself a "
            "proposed action with its own preview and its own approval"
        ),
        kind=ActionKind.ROLLBACK,
        args={"generation": generation.number},
    )


def r3_fingerprint(runner: Runner, files: Files) -> Fingerprint:
    """What the system was when the preview was taken.

    Both halves matter. The store path answers "is this still the system the
    diff was computed against", and the generation number answers "has anything
    switched since" even in the case where a switch landed on the same closure.
    An unreadable half is left empty, which will not match a later read - a
    proposal previewed against a system nobody could describe must not become
    appliable.
    """
    system = current_system(files)
    generation = current_generation(runner)
    number = generation.number if generation is not None else 0
    value = f"{system}|gen:{number}" if system and number else ""
    return Fingerprint(
        value=value,
        describes="the running system's store path and generation number",
    )


def activate_preview(
    plan: Plan, runner: Runner, files: Files
) -> tuple[Preview, Fingerprint, Reversal]:
    """What switching to a freshly built configuration would change."""
    toplevel = str(plan.args["toplevel"])
    rev = str(plan.args.get("rev", ""))
    repo = str(plan.args.get("repo", ""))
    system = current_system(files)
    generation = current_generation(runner)
    fingerprint = r3_fingerprint(runner, files)
    reversal = _reversal_to(generation, leaving="is running")

    if not system:
        return (
            Preview(
                kind=PreviewKind.NONE,
                headline=plan.summary,
                label="the running system could not be identified",
                available=Availability.unavailable(
                    f"{CURRENT_SYSTEM} does not resolve, so there is nothing to "
                    "compare the built configuration against"
                ),
            ),
            fingerprint,
            reversal,
        )

    lines: list[str] = []
    if generation is not None:
        lines.append(
            f"now:  generation {generation.number}"
            + (f" ({generation.date})" if generation.date else "")
            + f" -> {system}"
        )
    else:
        lines.append(f"now:  {system} (generation number unreadable)")
    lines.append(f"next: {toplevel}")
    if rev:
        lines.append(f"built from: {rev}" + (f" in {repo}" if repo else ""))
    lines.append("")

    diff, changed, caveat = closure_diff(runner, system, toplevel)
    if caveat:
        return (
            Preview(
                kind=PreviewKind.NONE,
                headline=plan.summary,
                label="the closure diff could not be produced",
                lines=lines,
                available=Availability.unavailable(
                    f"nix could not diff the two closures: {caveat}"
                ),
            ),
            fingerprint,
            reversal,
        )
    if not changed:
        lines.append(
            "no closure change: the built configuration is byte-identical to the "
            "system already running. Activating it would create a new generation "
            "that changes nothing."
        )
    else:
        lines.append("closure diff (what packages change):")
        lines.extend(f"  {line}" for line in diff)
    lines.append("")
    lines.append(_no_execution_note())
    lines.append("")
    lines.extend(_command_lines(plan))

    return (
        Preview(
            kind=PreviewKind.SIMULATION,
            headline=plan.summary,
            label=(
                "nix's own comparison of the running closure against the built "
                "one. It is a complete answer about PACKAGES and says nothing "
                "about what the activation scripts in this configuration will do."
            ),
            lines=lines,
        ),
        fingerprint,
        reversal,
    )


def rollback_preview(
    plan: Plan, runner: Runner, files: Files
) -> tuple[Preview, Fingerprint, Reversal]:
    """What returning to an earlier generation would change."""
    number = int(str(plan.args["generation"]))
    toplevel = str(plan.args["toplevel"])
    system = current_system(files)
    generation = current_generation(runner)
    fingerprint = r3_fingerprint(runner, files)
    # The inverse of a rollback is going back to where we are leaving from, which
    # is itself a generation - so it is the same verb with a different number.
    reversal = _reversal_to(generation, leaving="is running")

    lines: list[str] = [
        f"now:  generation {generation.number if generation else '?'} -> "
        f"{system or 'unreadable'}",
        f"back to: generation {number} -> {toplevel}",
        f"link: {generation_link(number)}",
        "",
    ]
    if not system:
        return (
            Preview(
                kind=PreviewKind.NONE,
                headline=plan.summary,
                label="the running system could not be identified",
                lines=lines,
                available=Availability.unavailable(
                    f"{CURRENT_SYSTEM} does not resolve, so there is nothing to "
                    f"compare generation {number} against"
                ),
            ),
            fingerprint,
            reversal,
        )
    diff, changed, caveat = closure_diff(runner, system, toplevel)
    if caveat:
        return (
            Preview(
                kind=PreviewKind.NONE,
                headline=plan.summary,
                label="the closure diff could not be produced",
                lines=lines,
                available=Availability.unavailable(
                    f"nix could not diff the two closures: {caveat}"
                ),
            ),
            fingerprint,
            reversal,
        )
    if not changed:
        lines.append(
            f"no closure change: generation {number} holds the same closure as "
            "the running system, so this would move the profile without changing "
            "any package."
        )
    else:
        lines.append(f"closure diff (running system -> generation {number}):")
        lines.extend(f"  {line}" for line in diff)
    lines.append("")
    lines.append(
        "a rollback restores the CONFIGURATION, not the data anything did while "
        "the newer one was running: state a service migrated, files it wrote and "
        "packages that were garbage collected do not come back."
    )
    lines.append("")
    lines.append(_no_execution_note())
    lines.append("")
    lines.extend(_command_lines(plan))

    return (
        Preview(
            kind=PreviewKind.SIMULATION,
            headline=plan.summary,
            label=(
                "nix's own comparison of the running closure against the "
                "generation being restored. This generation has run on this "
                "machine before, which is what makes it a rollback rather than a "
                "new configuration."
            ),
            lines=lines,
        ),
        fingerprint,
        reversal,
    )
