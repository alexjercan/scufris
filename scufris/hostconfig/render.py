"""One configuration change as plain text, for an agent to relay verbatim."""

from __future__ import annotations

from .models import ConfigChange


def render_change(change: ConfigChange) -> str:
    """One configuration change as plain text, for an agent to relay verbatim.

    Everything an operator needs before they open the approval: which revision,
    what its commit says, whether it is merged, what is NOT in the build, and
    where the change is in its life. The closure diff is not here - it belongs to
    the host action's own preview, which is rendered by
    ``host_actions.render_action``.
    """
    resolved = change.resolved
    lines = [
        f"nixos config change {change.id}",
        f"  repo:     {resolved.repo}",
        f"  ref:      {resolved.ref} @ {resolved.rev[:12]}",
        f"  commit:   {resolved.subject or '(no subject)'}",
        f"  host:     nixosConfigurations.{change.attr}",
        f"  state:    {change.state}",
    ]
    if resolved.merged is False:
        lines.append(
            f"  NOTE:     {resolved.rev[:12]} is not in "
            f"{resolved.head_branch or 'the checkout'} yet - merging it back is a "
            "separate act, and until it happens a later change branched from "
            f"{resolved.head_branch or 'the checkout'} will not contain it"
        )
    if resolved.uncommitted:
        shown = ", ".join(resolved.uncommitted[:10])
        more = (
            f" and {len(resolved.uncommitted) - 10} more"
            if len(resolved.uncommitted) > 10
            else ""
        )
        lines.append(
            f"  NOTE:     {len(resolved.uncommitted)} uncommitted file(s) in that "
            f"working tree are NOT in this build ({shown}{more}): the build takes "
            "the tree from the commit, on purpose"
        )
    if change.toplevel:
        lines.append(f"  built:    {change.toplevel}")
    if change.action_id:
        lines.append(f"  action:   {change.action_id} (awaiting the operator)")
    if change.error:
        lines.append(f"  ERROR:    {change.error}")
    if change.log_tail:
        tail = change.log_tail.strip().splitlines()[-20:]
        lines.append("  last lines of the build log:")
        lines.extend(f"    {line}" for line in tail)
    return "\n".join(lines)
