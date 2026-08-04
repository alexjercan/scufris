# Add plugin manifests discovery and health reporting

- PRIORITY: 0
- TAGS: feature, backlog, plugins, mcp, backend
- ACTIVITY: PLANNING
- GATES: -
- RESOLUTION: -

## Story

As an operator, I want Scufris to discover and explain installed plugins, so
that I can see what capabilities are available, what configuration they need,
whether they are healthy, and which agents may use them.

## Steps

- [ ] Record the plugin manifest, discovery, process, and health boundary in
      `DECISION.md`; this task owns that design rather than inheriting it from
      the reusable agent-preset spike.
- [ ] Implement versioned plugin-manifest parsing, validation, conflict
      detection, and deterministic discovery from configured directories.
- [ ] Model tools, resources, prompts, config schema, secret references,
      declared capabilities, transport/process command, health probe, and
      optional UI/artifact contributions.
- [ ] Add lifecycle management for out-of-process plugin transports with
      readiness, timeout, restart, failure isolation, and clean shutdown.
- [ ] Adapt existing MCP server discovery into the plugin catalog without
      removing role-scoped tool filtering.
- [ ] Add read-only catalog and health APIs plus an operator UI showing
      installed, unavailable, unhealthy, incompatible, and disabled states.
- [ ] Add example and hostile manifests covering version skew, duplicate IDs,
      unsafe paths, missing secrets, excessive output, and unhealthy servers.

## Definition of Done

- Valid plugins are discovered deterministically and invalid manifests produce
  actionable diagnostics (test: `test_plugin_manifest_discovery_and_health`).
- Plugin failure cannot crash the Scufris server or expose another plugin's
  configuration (test: `test_plugin_process_failure_is_isolated`).
- Existing role-scoped MCP tools retain their current visibility boundaries
  (test: `test_plugin_catalog_preserves_agent_tool_scoping`).
- A sample out-of-process plugin works end to end
  (cmd: `python examples/plugin_catalog.py`).

## Notes

- Epic: 20260729-102204.
- Depends on: 20260729-102147.
- Re-plan this task and accept its manifest/process-boundary decision before
  implementation when the plugin platform is scheduled. 20260729-102205 only
  defines how presets reference capabilities that already exist.
- Initial plugins should be declarative wrappers around processes/MCP, not
  arbitrary imported Python modules.
