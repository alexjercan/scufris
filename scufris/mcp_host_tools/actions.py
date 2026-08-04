"""The propose-only host action tools. An agent may ASK for a change, never make one.

There is deliberately no approve tool here, and there never will be: an approval
is an operator act, gated on a real session by the middleware
(`auth.OPERATOR_ONLY_PATTERN`). The absence is enforced twice - by there being no
tool, and by the HTTP endpoint refusing the machine bearer token these
subprocesses hold. `tests/test_mcp_server.py` asserts the absence, so a future
convenience tool cannot quietly appear.
"""

from __future__ import annotations

import json
import os

from ..mcp_common import _api_call


def propose_host_action(
    action: str, unit: str = "", days: int = 0, generation: int = 0
) -> str:
    """Propose a privileged change to THIS host, for the operator to approve.

    Nothing happens when you call this. It returns a PREVIEW - what would
    change, what else it reaches, and how it could be undone - and leaves the
    action waiting for the operator. You cannot approve it; only a human with a
    dashboard session can.

    Use it for "restart that service" and "clean up disk space" instead of
    trying to run systemctl or nix-collect-garbage in the shell, which will fail:
    those need root, and the only route to root on this box is this proposal.

    `action` is one of: unit_start, unit_stop, unit_restart, unit_reload (pass
    `unit`, e.g. "nginx" or "nginx.service"), gc_store (no arguments),
    gc_older_than (pass `days`), or rollback (pass `generation` - the number from
    the generation list, which returns the whole system to that configuration).

    Changing the NixOS CONFIGURATION is not here: use `propose_nixos_change`,
    which builds a commit first. `activate` is refused on this path outright,
    because what gets activated must be something the server built from an
    identified revision rather than a path handed to it.

    Show the operator the preview text verbatim rather than summarising it - the
    label saying whether it is a simulation or a statement of current state is
    part of the answer, not decoration.
    """
    args: dict[str, object] = {}
    if unit:
        args["unit"] = unit
    if days:
        args["days"] = days
    if generation:
        args["generation"] = generation
    # Name ourselves in the audit. The API derives the ACTOR from the credential
    # (this subprocess presents the machine token, so it is recorded as an agent
    # whatever it claims here); this only says WHICH agent, so a record names
    # something more useful than "an agent".
    answer = _api_call(
        "POST",
        "/api/host/actions",
        body={
            "kind": action,
            "args": args,
            "agent": os.environ.get("SCUFRIS_AGENT_ID", "orchestrator"),
        },
    )
    return _render_host_action(answer)


def _render_host_action(answer: str) -> str:
    """Render a host action response as the operator-facing text.

    The tool asks the model to show the preview verbatim, so it hands it prose
    rather than JSON to paraphrase - the label saying whether this is a
    simulation or a statement of current state is part of the answer, and a model
    summarising a JSON blob is exactly where that gets dropped.

    A non-JSON answer is an `error: ...` line from `_api_call`; pass it through
    unchanged rather than turning a diagnosable failure into a parse error.
    """
    from scufris_hostctl import HostActionRecord, render_action

    try:
        payload = json.loads(answer)
    except ValueError:
        return answer
    try:
        record = HostActionRecord.model_validate(payload)
    except Exception:  # noqa: BLE001 - an unexpected shape is still an answer
        return answer
    return (
        f"{render_action(record)}\n\n"
        "This is a PROPOSAL. Nothing has happened yet, and you cannot approve "
        "it - the operator must, in the dashboard. Show them the preview above "
        "as it is written."
    )


def host_action_status(action_id: str = "") -> str:
    """What has happened to a proposed host action (or all of them).

    Use it after `propose_host_action` to tell the operator whether their
    approval has landed and what the result was. With no id, lists the queue.
    """
    path = f"/api/host/actions/{action_id}" if action_id else "/api/host/actions"
    return _api_call("GET", path)


def propose_nixos_change(ref: str = "", repo: str = "", attr: str = "") -> str:
    """Build a COMMITTED NixOS configuration and propose activating it.

    Use this after the configuration repository has actually been changed and
    committed - which is ordinary project work, not a host action: open a
    worktree on that project, edit it, commit on a branch, review it. Then name
    that branch here.

    `ref` is a branch, tag or commit (default: HEAD of that working tree). `repo`
    is the configuration repository or one of its worktrees (default: the
    configured one). `attr` is the nixosConfiguration to build (default: this
    machine's hostname).

    What happens: the ref is resolved to a commit, that commit is BUILT as the
    operator (not as root), and if it builds, the activation is proposed for the
    operator to approve - with a closure diff against the running system as its
    preview. Nothing is activated by this call, and you cannot approve it.

    The build takes the tree from the COMMIT, so uncommitted edits are not in it.
    A build failure ends here: the log comes back and no proposal is created.
    Building can take a long time; poll `nixos_change_status`.
    """
    body: dict[str, object] = {
        "agent": os.environ.get("SCUFRIS_AGENT_ID", "orchestrator"),
    }
    if ref:
        body["ref"] = ref
    if repo:
        body["repo"] = repo
    if attr:
        body["attr"] = attr
    answer = _api_call("POST", "/api/host/config/changes", body=body)
    return _render_config_change(answer)


def nixos_change_status(change_id: str = "") -> str:
    """How a proposed NixOS configuration change is doing (or all of them).

    Use it after `propose_nixos_change`: while the state is `building` the build
    is still running, `failed` carries the build log, and `proposed` means there
    is a host action waiting for the operator - read that with
    `host_action_status` to show them the closure diff.
    """
    path = (
        f"/api/host/config/changes/{change_id}"
        if change_id
        else "/api/host/config/changes"
    )
    answer = _api_call("GET", path)
    return _render_config_change(answer) if change_id else answer


def _render_config_change(answer: str) -> str:
    """Render a config-change response as the operator-facing text.

    Same reasoning as `_render_host_action`: the notes about what is NOT in the
    build and whether the revision is merged are part of the answer, and a model
    paraphrasing JSON is exactly where they get dropped.
    """
    from scufris_hostctl import ConfigChange, render_change

    try:
        payload = json.loads(answer)
    except ValueError:
        return answer
    try:
        change = ConfigChange.model_validate(payload)
    except Exception:  # noqa: BLE001 - an unexpected shape is still an answer
        return answer
    text = render_change(change)
    if change.action_id:
        return (
            f"{text}\n\nThe activation is PROPOSED as host action "
            f"{change.action_id}. Read it with host_action_status and show the "
            "operator its preview verbatim; only they can approve it."
        )
    return text


def host_action_audit(limit: int = 20) -> str:
    """The record of privileged host actions: requested, refused, approved, applied.

    Written by the root helper itself, so it is the authoritative answer to
    "what has been done to this box", including actions this agent never saw.
    """
    return _api_call("GET", f"/api/host/audit?limit={max(1, min(500, limit))}")
