"""Read-only discovery of a project's per-project SKILLS and custom TOOLS.

A project can define, in its own working tree, the skills and MCP tools its
agents get. This module DISCOVERS them for display - it never writes, and never
executes any discovered command. It mirrors the ``read_project_tasks`` pattern
in ``projects.py``: cwd-scoped, guarded by directory/file existence, tolerant of
missing or malformed input (a bad file is skipped with a log line, never
raises), and returns plain records.

Discovery is PROVIDER-AWARE: the source paths are chosen by the agent's backend
(``canonical_backend``), because Claude Code and codex use different on-disk
conventions. Only PROJECT-tree sources are scanned (never the operator's global
``~/.claude`` / ``~/.codex``), matching the per-project scope.
"""

from __future__ import annotations

import json
import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from .config import canonical_backend

logger = logging.getLogger(__name__)


class ProjectSkill(BaseModel):
    """One project-defined skill (a ``SKILL.md`` recipe an agent can be steered
    toward). ``source`` is the file path relative to the project cwd."""

    name: str
    description: str = ""
    source: str


class ProjectTool(BaseModel):
    """One project-defined MCP server (the concrete per-project custom-tool
    surface). ``kind`` is the transport ("stdio"/"http"/"sse"/"ws", or "" when
    unknown); ``description`` is a short best-effort summary (command or url);
    ``source`` is the config file path relative to the project cwd."""

    name: str
    description: str = ""
    source: str
    kind: str = ""


class ProjectCapabilities(BaseModel):
    """The read-only capability surface of a project: its skills and tools."""

    skills: list[ProjectSkill] = []
    tools: list[ProjectTool] = []


@dataclass(frozen=True)
class ProviderSources:
    """Where a given backend keeps its per-project skills and MCP config.

    ``skill_dirs`` hold ``<name>/SKILL.md`` recipes; ``mcp_json_files`` are JSON
    with a top-level ``mcpServers`` object (Claude Code); ``mcp_toml_files`` are
    TOML with ``[mcp_servers.<name>]`` tables (codex). All paths are relative to
    the project cwd. An unknown backend maps to the empty sources -> no
    discovery."""

    skill_dirs: tuple[str, ...] = ()
    mcp_json_files: tuple[str, ...] = ()
    mcp_toml_files: tuple[str, ...] = ()


# The provider registry, keyed by CANONICAL backend. Data-driven so a new
# provider is one entry, not new code (GOAL.md done-item 5).
_PROVIDER_SOURCES: dict[str, ProviderSources] = {
    "claude": ProviderSources(
        skill_dirs=(".claude/skills",),
        mcp_json_files=(
            ".mcp.json",
            ".claude/settings.json",
            ".claude/settings.local.json",
        ),
    ),
    "codex": ProviderSources(
        skill_dirs=(".codex/skills",),
        mcp_toml_files=(".codex/config.toml",),
    ),
}


def _sources_for(backend: str) -> ProviderSources:
    return _PROVIDER_SOURCES.get(canonical_backend(backend), ProviderSources())


def _parse_frontmatter(text: str) -> dict[str, str]:
    """The ``---``-delimited YAML head of a SKILL.md, as flat single-line keys.

    A minimal parser (no PyYAML dep): everything between the first two ``---``
    fences, ``key: value`` split on the first colon. Enough for a skill's
    single-line ``name`` and ``description``; anything fancier is ignored, not
    an error. No fence -> ``{}``."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep and key.strip():
            fields[key.strip()] = value.strip()
    return fields


def read_project_skills(cwd: str, backend: str) -> list[ProjectSkill]:
    """The project's ``SKILL.md`` skills for ``backend``, sorted by name.

    Scoped to the backend's skill dirs under ``cwd``: a missing dir yields
    nothing (no walk upward, unlike tatr's ``-r``). Tolerant - an unreadable
    file is skipped with a log line. ``name`` defaults to the skill directory
    when the frontmatter omits it; ``description`` defaults to empty."""
    root = Path(cwd)
    skills: list[ProjectSkill] = []
    for rel_dir in _sources_for(backend).skill_dirs:
        skill_dir = root / rel_dir
        if not skill_dir.is_dir():
            continue
        for skill_md in skill_dir.glob("*/SKILL.md"):
            try:
                text = skill_md.read_text()
            except OSError as exc:
                logger.warning("read_project_skills: cannot read %s: %s", skill_md, exc)
                continue
            fields = _parse_frontmatter(text)
            skills.append(
                ProjectSkill(
                    name=fields.get("name") or skill_md.parent.name,
                    description=fields.get("description", ""),
                    source=str(skill_md.relative_to(root)),
                )
            )
    skills.sort(key=lambda s: s.name.lower())
    return skills


def _tool_from_spec(name: str, spec: object, source: str) -> ProjectTool:
    """One ``ProjectTool`` from an MCP server spec (a JSON object or TOML table).

    ``kind`` prefers an explicit ``type``, else infers "stdio" from a
    ``command`` or "http" from a ``url``. ``description`` is the command (plus
    its first arg) or the url, best-effort - never the ``env`` (may hold
    secrets)."""
    kind = ""
    description = ""
    if isinstance(spec, dict):
        declared = spec.get("type")
        if isinstance(declared, str) and declared.strip():
            kind = declared.strip()
        command = spec.get("command")
        url = spec.get("url")
        if isinstance(command, str) and command:
            if not kind:
                kind = "stdio"
            args = spec.get("args")
            if isinstance(args, list) and args:
                description = f"{command} {args[0]}"
            else:
                description = command
        elif isinstance(url, str) and url:
            if not kind:
                kind = "http"
            description = url
    return ProjectTool(name=name, description=description, source=source, kind=kind)


def _load_mcp_servers(path: Path, loader: str) -> dict[str, object]:
    """The ``mcpServers`` / ``[mcp_servers]`` mapping from a config file.

    ``loader`` is "json" or "toml". A MISSING file is silent (return ``{}``); a
    malformed one is logged and skipped. Never raises."""
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("project tools: cannot read %s: %s", path, exc)
        return {}
    try:
        data: object = json.loads(raw) if loader == "json" else tomllib.loads(raw)
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        logger.warning("project tools: cannot parse %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    key = "mcpServers" if loader == "json" else "mcp_servers"
    servers = data.get(key)
    return servers if isinstance(servers, dict) else {}


def read_project_tools(cwd: str, backend: str) -> list[ProjectTool]:
    """The project's custom MCP servers for ``backend``, sorted by name.

    Merges the backend's JSON and TOML config files in registry order,
    de-duplicating by server name (the FIRST file to declare a name wins).
    Tolerant - a missing or malformed file contributes nothing, never raises."""
    root = Path(cwd)
    sources = _sources_for(backend)
    tools: list[ProjectTool] = []
    seen: set[str] = set()
    files = [(rel, "json") for rel in sources.mcp_json_files] + [
        (rel, "toml") for rel in sources.mcp_toml_files
    ]
    for rel, loader in files:
        source = rel
        for name, spec in _load_mcp_servers(root / rel, loader).items():
            if name in seen:
                continue
            seen.add(name)
            tools.append(_tool_from_spec(name, spec, source))
    tools.sort(key=lambda t: t.name.lower())
    return tools


def read_project_capabilities(cwd: str, backend: str) -> ProjectCapabilities:
    """A project's read-only capability surface: its skills and tools together.

    The single entry point the HTTP endpoint calls."""
    return ProjectCapabilities(
        skills=read_project_skills(cwd, backend),
        tools=read_project_tools(cwd, backend),
    )
