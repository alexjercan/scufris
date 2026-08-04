# Understanding: reduce Telegram to a chat and host-approval surface

## What changes

Telegram stops depending on the orchestrator stack. Agent operations are
removed, not stubbed.

## Surfaces

- `scufris/telegram/wiring.py` - the coupling, at MODULE scope.
- Ten modules under `scufris/telegram/`: `api`, `approvals`, `bot`, `contracts`,
  `orchestrator`, `render`, `text`, `turn`, `wiring`, `__init__`.
- `scufris/telegram/orchestrator.py` is the obvious casualty; `turn.py` and
  `contracts.py` need reading before assuming.

## Data and interfaces

`wiring.py` imports at module scope, confirmed by reading it:

```
from ..agent_diagnostics import (...)
from ..agent_store import ORCHESTRATOR_ID, AgentStore
from ..config import Settings
from ..env_bridge import ensure_den_path
from ..health import AgentHealth
from ..mcp_models import AgentTool
from ..orchestrator import OrchestratorTurnService
```

Module scope is the whole problem. Deleting those modules in Lane 8 makes
Telegram fail to IMPORT - it does not degrade into a polite refusal. That is why
this task exists and why it is scheduled before any deletion.

## Sketches

```
  BEFORE                        AFTER
    telegram                      telegram
      |- conversation               |- conversation
      |- host approvals             |- host approvals
      |- agent ops  --> orchestrator, agent_store,
                        health, agent_diagnostics, mcp_models
                        ^^^ removed, not stubbed
```

## Shape

Remove rather than stub. A command that answers "not available" is a surface to
keep working, to test and to delete later; the point of the reduction is that
the package gets smaller.

The operator uses this surface daily, so conversation and host approvals must
keep working throughout - this is not a "Telegram is broken for a while" task.

## Consequences and open questions

- **Lane placement is the known-weak decision.** Its only HARD constraint is
  that it precede the Lane 5 deletions. It sits in Lane 2 for coherence with the
  approval card, and the epic already records that it should be MOVED rather
  than allowed to become Lane 5's blocker if Lane 2 slips. Worth re-checking at
  planning rather than inheriting.
- **Open:** whether the reduction should wait for the approval decoupling at
  all. It depends on it here because the card's interface changes, but the
  agent-op removal is independent of that and could land first. Splitting into
  "remove agent ops" and "rewire the card" would decouple this from Lane 2's
  riskiest task. Leaning: split if the decoupling looks like it will overrun.
- **Open:** what happens to `examples/telegram_approval.py` and
  `examples/telegram_bot.py`. Neither is on `OFFLINE`. The carve task needs one
  of them - or a new one - to be offline, and this task is where the surface it
  demonstrates stops changing.
