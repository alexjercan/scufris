# Review: EPIC close-out and architecture map

- TASK: 20260729-124655
- BRANCH: docs/epic-architecture

This container's IMPLEMENTATION was reviewed eight times over, once per child task
(each child's own REVIEW.md: 21 rounds, 72 findings, 5 BLOCKER, 14 MAJOR in total).
This record covers only what this branch changed: `ARCHITECTURE.md`, the close-out
note in `TASK.md`, the status flip to CLOSED, and one stale citation in `AGENTS.md`.

## Round 1

- VERDICT: APPROVE
- REVIEWER: in-session self-review. Recorded as the review skill's trivial-diff
  carve-out: the branch touches no code, adds no test surface, and changes no
  behavior. What it CAN get wrong is describing a system that does not exist, so the
  review was a claim-by-claim check of the document against the code it describes,
  not a read of the prose.

### What was verified rather than taken on trust

Every structural claim in `ARCHITECTURE.md` was checked against the file it names:

- The verb set and risk mapping against `hostd/actions.py` (`ActionKind`,
  `RiskClass`, `RISK_OF`), including that R4 has no member.
- The two-step R3 plans against `build_plan`'s `activate` and `rollback` steps
  (`nix-env --set` / `--switch-generation`, then `_switch_step` via `systemd-run`).
- The four apply refusals and the proposal state machine against `hostd/engine.py`
  and `hostd/protocol.py` (`ProposalState`, `ErrorCode`).
- The "10 minute window" against `engine.DEFAULT_TTL_SECONDS = 600.0` rather than
  against the phrase used in the task records.
- The confirmation rule and its R1 carve-out against
  `host_actions.confirmation_for`.
- `decidable()` and `live_for_agent()` against `host_approvals.py`.
- The operator-only surface against `auth.OPERATOR_ONLY_PATTERN`, including that
  `/api/host/digests/run` is in it (the doc's boundary table says "or run the checks
  on demand" because of that, not by guess).
- The public allowlist against `auth.PUBLIC_PATHS` / `PUBLIC_STATIC_PATHS`.
- The audience split against `enums.audience_for` and `agent.scufris_mcp_servers`,
  and the tool names against the actual function names in `mcp_host_tools.py`.
- The secret-stripping claim against `config.SECRET_ENV_VARS`.
- Every path in the file map, every example in the proofs table, and both flake
  outputs (`nixosModules.hostd`, `.#hostd-vm-test`) confirmed to exist.

### Findings, all fixed on this branch

- **MAJOR - a citation copied from `AGENTS.md` was wrong.** The doc said
  `tests/test_mcp_server.py` asserts the absence of an approve tool. It does not:
  that file only tests `apply_disabled_tools`. The assertion is
  `tests/test_host_mcp_server.py::test_the_agent_has_no_tool_that_approves_a_host_action`.
  Found by grepping for the test rather than trusting the sentence that named it.
  Fixed in the new doc AND in `AGENTS.md:489`, which is where the error came from -
  a wrong pointer to the test that proves the epic's central refusal is worth more
  than a typo, because the next reader checks the empty file and concludes nothing
  proves it.
- **MAJOR - the first draft drew Telegram as inbound HTTP.** `PHONE --> API` implies
  a webhook the deployment does not have. `telegram.py` owns a `getUpdates` long
  poll and there is no public webhook, so the bot is a component INSIDE the user
  unit that talks outward. Redrawn with the bot as its own node, an outbound-only
  edge, and its own row in the boundary table naming the allowlist as the
  credential. A trust-boundary diagram that invents an inbound port is worse than
  no diagram.
- **MINOR - "systemd USER unit" was stated as if it were the only shape.**
  `nix/scufris-service.nix` emits a NixOS system unit with `DynamicUser` OR a
  home-manager `systemd.user` unit; this host deploys the latter. The file map now
  says both and names which one is deployed, because the whole
  secret-stripping argument depends on the app running as the operator.
- **NIT - angle brackets inside mermaid labels.** `<toplevel>`, `<unit>` and
  `<chat_id>` inside mermaid node/message text render as unknown HTML tags and
  disappear. Replaced with `[...]` placeholders inside the diagrams; the ASCII
  blocks and prose keep the angle brackets, where they are safe.

### Deliberately not done

- No `NOTES.md`: this branch ships no change to the software, and `ARCHITECTURE.md`
  IS the design record.
- The eight Manual Acceptance items were NOT ticked. The operator closed the epic by
  declaration; ticking items that need a deployment, a phone and a week of digests
  would have turned a decision into fabricated evidence. The close-out note says so
  in as many words.
