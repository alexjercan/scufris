# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The built-in `scufris` MCP server (host/observe tools and its tool-steering
  preamble) is now ORCHESTRATOR-ONLY: it is registered for the landing
  orchestrator's turns only, not for every agent. Regular project agents no longer
  receive the scufris tools and draw their tools from their own project
  config/skills. This threads an `is_orchestrator` flag through the backend
  `stream` path; operator-declared `mcp_servers` still apply to every agent.

### Added

- Settings page: an interactive "try it" runner on each enabled tool card - reveal
  a form generated from the tool's parameter schema, confirm, and run one MCP tool
  in isolation with its result rendered inline, without a chat turn. Backed by a new
  `POST /api/agent/tools/{name}/run` endpoint that runs a single scufris tool
  in-process (bypassing the agent) and refuses a disabled tool (403), an unknown tool
  (404), or bad args (422). The tools listing (`GET /api/agent/tools`) now also
  exposes each tool's typed parameter schema.
