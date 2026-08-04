"""The unprivileged client that drives the host helper.

`scufris_host` reads the machine and needs no privilege. `scufris_hostd` runs as
root in another process and owns the verbs, the proposals and the audit log.
THIS package is the client between them: it builds an action, gets a preview,
holds it for an operator decision, dispatches the approved action over the
socket, watches the result, and records what happened.

| Module | Role |
|---|---|
| `actions` | the app-side record: the durable decision journal, `confirmation_for`, and `render_action` - the one renderer both operator surfaces use |
| `approvals` | the decision seam: approve / deny / cancel / revert / `decidable()`. `apply` is called from exactly one place |
| `client` | the socket itself: connect, one authenticated request, read frames. An apply is a stream that can be cut |
| `hostconfig` | the R3 NixOS configuration change flow: resolve a ref, build it as this user, propose the activation, roll a generation back |
| `models` | the two tables this package owns, declared against `scufris_core.Base`. PRIVATE |

**This module is the whole public surface.** A caller outside the package
imports `scufris_hostctl`, never `scufris_hostctl.client` or
`scufris_hostctl.models`, and `test_no_package_imports_a_sibling_private_module`
enforces that for every workspace member.

The two row classes are deliberately absent from it. What a caller gets is
`HostActionStore` and `ConfigChangeStore`; the tables are how they are stored.
"""

from __future__ import annotations

from .actions import (
    MAX_ACTIONS,
    AlreadyDecided,
    Confirmation,
    ConfirmationStyle,
    Decision,
    HostActionRecord,
    HostActionStore,
    UnknownAction,
    confirmation_for,
    render_action,
)
from .approvals import (
    CannotPropose,
    CannotUndo,
    ConfirmationRequired,
    Hook,
    HostApprovalService,
    NoLiveRun,
    NotApplied,
    ProposalExpired,
    RecordList,
    decision_message,
)
from .client import (
    CONNECT_TIMEOUT,
    READ_LIMIT,
    HostApplyBus,
    HostApplyError,
    HostApplyEvent,
    HostApplyOutput,
    HostApplyResult,
    HostdClient,
    HostdError,
    HostdUnavailable,
    HostSupervisor,
    host_supervisor,
)
from .hostconfig import (
    ChangeInFlight,
    ChangeState,
    ConfigBuildBus,
    ConfigBuildDone,
    ConfigBuildError,
    ConfigBuildEvent,
    ConfigBuildOutput,
    ConfigChange,
    ConfigChangeBuilder,
    ConfigChangeRefused,
    ConfigChangeService,
    ConfigChangeStore,
    ConfigSupervisor,
    NoRunningBuild,
    Proposer,
    Resolved,
    UnknownChange,
    build_argv,
    check_attr,
    config_supervisor,
    default_attr,
    flake_url,
    render_change,
    resolve,
    toplevel_from,
)

__all__ = [
    "CONNECT_TIMEOUT",
    "MAX_ACTIONS",
    "READ_LIMIT",
    "AlreadyDecided",
    "CannotPropose",
    "CannotUndo",
    "ChangeInFlight",
    "ChangeState",
    "ConfigBuildBus",
    "ConfigBuildDone",
    "ConfigBuildError",
    "ConfigBuildEvent",
    "ConfigBuildOutput",
    "ConfigChange",
    "ConfigChangeBuilder",
    "ConfigChangeRefused",
    "ConfigChangeService",
    "ConfigChangeStore",
    "ConfigSupervisor",
    "Confirmation",
    "ConfirmationRequired",
    "ConfirmationStyle",
    "Decision",
    "Hook",
    "HostActionRecord",
    "HostActionStore",
    "HostApplyBus",
    "HostApplyError",
    "HostApplyEvent",
    "HostApplyOutput",
    "HostApplyResult",
    "HostApprovalService",
    "HostSupervisor",
    "HostdClient",
    "HostdError",
    "HostdUnavailable",
    "NoLiveRun",
    "NoRunningBuild",
    "NotApplied",
    "ProposalExpired",
    "Proposer",
    "RecordList",
    "Resolved",
    "UnknownAction",
    "UnknownChange",
    "build_argv",
    "check_attr",
    "config_supervisor",
    "confirmation_for",
    "decision_message",
    "default_attr",
    "flake_url",
    "host_supervisor",
    "render_action",
    "render_change",
    "resolve",
    "toplevel_from",
]
