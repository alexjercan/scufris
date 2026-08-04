"""``scufris-hostd``: the privileged half of the host action framework.

This package runs as ROOT, in its own process, behind a unix socket. The app
never imports it to act - it imports it for types, and talks to the running
helper through ``scufris_hostctl.client``.

The contract:

    propose -> preview -> approve -> apply -> audit -> roll back

- ``actions``  - the verb set, which IS the risk taxonomy. R4 has no verb.
- ``nixos``    - R3: the configuration change, its closure-diff preview, and the
                 one thing that preview deliberately refuses to run.
- ``files``    - the filesystem seam, for the questions R3 asks of the store.
- ``preview``  - what the operator sees, and the honesty label saying what it is.
- ``engine``   - proposals, the four apply refusals, and the audit calls.
- ``audit``    - the root-owned, append-only, size-rotated record.
- ``executor`` - the one place a process is spawned, killable by process group.
- ``server``   - the socket, authenticated per frame.
"""

from .actions import (
    DENIED_UNIT_STEMS,
    PROTECTED_GENERATIONS,
    ActionKind,
    ActionRefused,
    Plan,
    RiskClass,
    Step,
    build_plan,
    normalise_unit,
)
from .audit import AuditEvent, AuditLog, AuditRecord, Requester
from .engine import HostdEngine, HostdRefusal
from .executor import Executor, FakeExecutor, run_action
from .files import DEFAULT_FILES, FakeFiles, Files, RealFiles
from .preview import Fingerprint, Preview, PreviewKind, Reversal, build_preview
from .protocol import (
    PROTOCOL_VERSION,
    ErrorCode,
    ProposalState,
    ProposalView,
    Request,
    ResultFrame,
    Verb,
    encode,
)
from .server import HostdServer

__all__ = [
    "ActionKind",
    "ActionRefused",
    "AuditEvent",
    "AuditLog",
    "AuditRecord",
    "DENIED_UNIT_STEMS",
    "ErrorCode",
    "Executor",
    "DEFAULT_FILES",
    "FakeExecutor",
    "FakeFiles",
    "Files",
    "Fingerprint",
    "HostdEngine",
    "HostdRefusal",
    "HostdServer",
    "PROTECTED_GENERATIONS",
    "PROTOCOL_VERSION",
    "Plan",
    "Preview",
    "PreviewKind",
    "ProposalState",
    "ProposalView",
    "Request",
    "Requester",
    "ResultFrame",
    "RealFiles",
    "Reversal",
    "RiskClass",
    "Step",
    "Verb",
    "build_plan",
    "build_preview",
    "encode",
    "normalise_unit",
    "run_action",
]
