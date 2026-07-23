"""Tests for read-only per-project skills+tools discovery (provider-aware).

Mirrors the ``read_project_tasks`` tests in test_projects.py: a temp project
tree, cwd-scoped discovery, tolerant of missing/malformed input (never raises).
"""

from __future__ import annotations

import json
from pathlib import Path

from scufris.project_capabilities import (
    ProjectCapabilities,
    read_project_capabilities,
    read_project_skills,
    read_project_tools,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _skill(root: Path, name: str, description: str) -> None:
    _write(
        root / ".claude" / "skills" / name / "SKILL.md",
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\nbody\n",
    )


# --- skills ---------------------------------------------------------------


def test_reads_claude_skills_frontmatter(tmp_path: Path) -> None:
    _skill(tmp_path, "deploy", "Ship the app to prod")
    _skill(tmp_path, "audit", "Review the diff")
    skills = read_project_skills(str(tmp_path), "claude")
    assert [s.name for s in skills] == ["audit", "deploy"]  # sorted by name
    deploy = next(s for s in skills if s.name == "deploy")
    assert deploy.description == "Ship the app to prod"
    assert deploy.source == ".claude/skills/deploy/SKILL.md"


def test_skill_name_defaults_to_dir_when_frontmatter_absent(tmp_path: Path) -> None:
    # No frontmatter at all -> name falls back to the directory, empty description.
    _write(tmp_path / ".claude" / "skills" / "bare" / "SKILL.md", "# just a heading\n")
    skills = read_project_skills(str(tmp_path), "claude")
    assert len(skills) == 1
    assert skills[0].name == "bare"
    assert skills[0].description == ""


def test_skills_empty_when_no_dir(tmp_path: Path) -> None:
    assert read_project_skills(str(tmp_path), "claude") == []


def test_codex_skills_use_codex_dir(tmp_path: Path) -> None:
    _write(
        tmp_path / ".codex" / "skills" / "imagegen" / "SKILL.md",
        "---\nname: imagegen\ndescription: make images\n---\n",
    )
    # A claude backend does NOT look in .codex/skills.
    assert read_project_skills(str(tmp_path), "claude") == []
    codex = read_project_skills(str(tmp_path), "codex")
    assert [s.name for s in codex] == ["imagegen"]
    assert codex[0].source == ".codex/skills/imagegen/SKILL.md"


# --- tools ----------------------------------------------------------------


def test_reads_mcp_json_and_settings_merged_and_deduped(tmp_path: Path) -> None:
    _write(
        tmp_path / ".mcp.json",
        json.dumps(
            {
                "mcpServers": {
                    "fs": {"command": "npx", "args": ["-y", "fs-server"]},
                    "remote": {"type": "http", "url": "https://example.com/mcp"},
                }
            }
        ),
    )
    # settings.json redefines `fs` (first file wins) and adds `db`.
    _write(
        tmp_path / ".claude" / "settings.json",
        json.dumps(
            {
                "mcpServers": {
                    "fs": {"command": "OTHER"},
                    "db": {"command": "pg-mcp"},
                }
            }
        ),
    )
    tools = read_project_tools(str(tmp_path), "claude")
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"fs", "remote", "db"}
    # .mcp.json wins for the duplicate `fs`.
    assert by_name["fs"].source == ".mcp.json"
    assert by_name["fs"].kind == "stdio"
    assert by_name["fs"].description == "npx -y"
    assert by_name["remote"].kind == "http"
    assert by_name["remote"].description == "https://example.com/mcp"
    assert by_name["db"].source == ".claude/settings.json"


def test_reads_codex_toml_tools(tmp_path: Path) -> None:
    _write(
        tmp_path / ".codex" / "config.toml",
        '[mcp_servers.fs]\ncommand = "fs-server"\nargs = ["--root", "."]\n'
        '\n[mcp_servers.remote]\ntype = "sse"\nurl = "https://x/sse"\n',
    )
    tools = read_project_tools(str(tmp_path), "codex")
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"fs", "remote"}
    assert by_name["fs"].kind == "stdio"
    assert by_name["fs"].source == ".codex/config.toml"
    assert by_name["remote"].kind == "sse"


def test_tools_empty_when_no_files(tmp_path: Path) -> None:
    assert read_project_tools(str(tmp_path), "claude") == []


# --- tolerance ------------------------------------------------------------


def test_malformed_files_are_tolerated(tmp_path: Path) -> None:
    _write(tmp_path / ".mcp.json", "{ not valid json")
    _write(tmp_path / ".codex" / "config.toml", "this = = = broken")
    # Malformed input yields no entries and does not raise.
    assert read_project_tools(str(tmp_path), "claude") == []
    assert read_project_tools(str(tmp_path), "codex") == []


def test_unknown_backend_discovers_nothing(tmp_path: Path) -> None:
    _skill(tmp_path, "deploy", "x")
    _write(tmp_path / ".mcp.json", json.dumps({"mcpServers": {"fs": {"command": "x"}}}))
    caps = read_project_capabilities(str(tmp_path), "mock")
    assert caps == ProjectCapabilities()


def test_read_project_capabilities_combines(tmp_path: Path) -> None:
    _skill(tmp_path, "deploy", "Ship it")
    _write(tmp_path / ".mcp.json", json.dumps({"mcpServers": {"fs": {"command": "x"}}}))
    caps = read_project_capabilities(str(tmp_path), "claude")
    assert [s.name for s in caps.skills] == ["deploy"]
    assert [t.name for t in caps.tools] == ["fs"]
