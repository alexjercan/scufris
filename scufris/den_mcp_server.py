"""Scufris `den` MCP server: the operator's the-den journal + macros life tools.

A SEPARATE MCP server (id ``den``) from the orchestrator agentic server
(``mcp_server``, id ``scufris``) and the sub-agent callback server
(``agent_mcp_server``, id ``agent``). Split out so the three concerns are
physically separate: an ORCHESTRATOR turn registers ``scufris`` + ``den``, and
this server never rides a sub-agent turn, so a project sub-agent can never reach
the operator's journal (the guarantee is "not registered", not a runtime
filter). See ``tasks/20260727-105609/DECISION.md``.

The tools read and update the operator's markdown journal ("the-den") through
the ``today`` CLI and look up food macros through the ``macros`` CLI. The den
directory is injected as ``SCUFRIS_DEN_PATH`` (by ``agent.scufris_mcp_servers``,
only onto this server). When the env is unset (no den configured) or the dir is
missing, the journal tools report a clear message and never shell out, so
scufris stays safe on a box without the-den. Every call passes an explicit
subcommand: bare ``today`` would open $EDITOR, which these tools must never do.

This module depends only on ``mcp_common`` (the ``_run`` shell wrapper + the
disabled-tools filter) and the ``today``/``macros`` CLIs - never on the
dashboard API, psutil, or the agent store - so it stays reusable standalone.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from .mcp_common import _run, apply_disabled_tools

logger = logging.getLogger(__name__)

mcp = FastMCP("den")


# --- the-den journal tools ----------------------------------------------------


def _den_path() -> str:
    """The configured the-den directory (``SCUFRIS_DEN_PATH``, injected by the
    dashboard onto this server), or ``""`` when no journal is configured.
    Env-read so the subprocess needs no Settings load."""
    import os

    return os.environ.get("SCUFRIS_DEN_PATH", "").strip()


def _journal(args: list[str]) -> str:
    """Run a `today` subcommand against the configured den and return bounded output.

    Validates the den is configured and present BEFORE shelling out: an unset
    ``SCUFRIS_DEN_PATH`` or a missing directory returns an ``error: ...`` message
    (never a traceback - the raw CLI raises on a bad den). Otherwise runs
    ``today --den <den> <args...>`` via ``_run`` (fixed argv, timeout, bounded).

    A leading ``~`` is expanded at use time, matching the repo convention for
    ``Path`` config (pydantic stores env paths verbatim; consumers ``expanduser``),
    so the ``~/personal/the-den`` form documented in ``.env.example`` works."""
    import os

    den = _den_path()
    if not den:
        return (
            "error: the-den journal is not configured (set SCUFRIS_DEN_PATH); the "
            "journal tools are unavailable on this host"
        )
    den = os.path.expanduser(den)
    if not os.path.isdir(den):
        return f"error: configured den path does not exist: {den}"
    return _run(["today", "--den", den, *args])


@mcp.tool()
def journal_show(offset: int = 0) -> str:
    """Read a day of the operator's the-den journal as JSON: date, title, habits
    (name+done), tasks and tomorrow's tasks (index+text+done), macros
    (protein/carbs/fat/calories) and logged weight.

    This is the PREFERRED, authoritative way to answer "what are today's tasks /
    habits / macros / weight" - use it INSTEAD of reading or grepping the journal's
    markdown files by hand. ``offset`` picks the day: 0 = today (default), -1 =
    yesterday, 1 = tomorrow-as-its-own-day. NOTE: notes are NOT included here - use
    ``journal_notes`` for those."""
    return _journal(["-N", str(offset), "show", "--json"])


@mcp.tool()
def journal_notes(tag: str = "") -> str:
    """List today's the-den notes as JSON (each note's text and its optional tag),
    optionally filtered to a single ``tag``.

    PREFER this over reading the journal markdown to answer "what are my notes" or
    "what notes did I tag <tag>". Notes are added with ``journal_add_note``."""
    args = ["note", "list"]
    if tag.strip():
        args += ["--tag", tag.strip()]
    args.append("--json")
    return _journal(args)


@mcp.tool()
def journal_add_task(text: str, tomorrow: bool = False) -> str:
    """Add a task to the operator's the-den journal and return the updated list as
    JSON. Set ``tomorrow=True`` to add it to Tomorrow's list instead of Today's
    (use this for "add a task for tomorrow").

    PREFER this over editing the journal markdown by hand."""
    if not text.strip():
        return "error: task text is required"
    args = ["task", "add", text]
    if tomorrow:
        args.append("--tomorrow")
    args.append("--json")
    return _journal(args)


@mcp.tool()
def journal_complete_task(index: int) -> str:
    """Toggle a Today task's done checkbox by its 1-based ``index`` (from
    ``journal_show``) and return the updated list as JSON. Use this for "check off"
    / "mark done" (calling it again un-checks the task).

    PREFER this over editing the journal markdown by hand."""
    return _journal(["task", "done", str(index), "--json"])


@mcp.tool()
def journal_remove_task(index: int, tomorrow: bool = False) -> str:
    """Remove a task by its 1-based ``index`` (from ``journal_show``) and return the
    updated list as JSON. Set ``tomorrow=True`` to remove from Tomorrow's list
    instead of Today's.

    PREFER this over editing the journal markdown by hand."""
    args = ["task", "rm", str(index)]
    if tomorrow:
        args.append("--tomorrow")
    args.append("--json")
    return _journal(args)


@mcp.tool()
def journal_toggle_habit(name: str) -> str:
    """Check or uncheck a habit by ``name`` (a leading emoji is optional, e.g. "Gym"
    matches "Gym") and return the updated habits as JSON. Use this for "check off
    gym" / "did my gym habit".

    PREFER this over editing the journal markdown by hand."""
    if not name.strip():
        return "error: habit name is required"
    return _journal(["habit", "toggle", name, "--json"])


@mcp.tool()
def journal_log_weight(value: str) -> str:
    """Log today's body weight to the-den (e.g. "80" or "80kg") and return the
    confirmation. Use this for "log 80kg" / "record my weight".

    PREFER this over editing the journal markdown by hand."""
    if not value.strip():
        return "error: weight value is required"
    return _journal(["weight", value])


@mcp.tool()
def journal_add_macros(row: str) -> str:
    """Append a macros row to today's the-den entry and return the updated daily
    aggregate (protein/carbs/fat/calories) as JSON. ``row`` is a CSV
    "what,protein,carbs,fat" (e.g. "eggs,20,2,15"). Use this for "log my macros" /
    "add a meal".

    PREFER this over editing the journal markdown by hand."""
    if not row.strip():
        return "error: macros row is required (what,protein,carbs,fat)"
    return _journal(["macros", "add", row, "--json"])


@mcp.tool()
def journal_add_note(text: str, tag: str = "") -> str:
    """Append a note to today's the-den entry and return the updated notes as JSON.
    Pass a single-word ``tag`` to mark it (e.g. tag="mood"). Use this for "add a
    note" / "jot this down".

    PREFER this over editing the journal markdown by hand."""
    if not text.strip():
        return "error: note text is required"
    args = ["note", "add", text]
    if tag.strip():
        args += ["--tag", tag.strip()]
    args.append("--json")
    return _journal(args)


# --- macros food-lookup tools -------------------------------------------------
#
# Wrap the `macros` CLI (github:alexjercan/macros.nvim), a food-macro lookup over a
# CSV database it resolves itself ($HOME/.local/share/nvim/macros.csv) - so, unlike
# the journal tools, there is no den/config knob: these just shell out via `_run`.
# The lookup output is the `<food> <amount><unit>,<protein>,<carbs>,<fat>` line, which
# is exactly the row `journal_add_macros` takes, so the two chain (look up a food's
# macros, then log them).


@mcp.tool()
def macros_lookup(query: str) -> str:
    """Look up a food's macros from the operator's macros database. ``query`` is a
    food plus an amount, e.g. "egg 2p" (2 pieces) or "chicken breast 100g". Returns a
    ``<food> <amount><unit>,<protein>,<carbs>,<fat>`` line (e.g. "egg 2pc,12,0,10").

    This is the PREFERRED way to answer "what are the macros for <food>". The result
    is exactly the row ``journal_add_macros`` accepts, so use it to log a meal: look
    up here, then pass the returned line to ``journal_add_macros``. If the food is
    unknown the CLI says so - use ``macros_search`` to find the right name."""
    if not query.strip():
        return 'error: a food query is required (e.g. "egg 2p")'
    return _run(["macros", query])


@mcp.tool()
def macros_search(query: str) -> str:
    """Fuzzy-search the foods available in the macros database, e.g. "chick" ->
    the matching food names.

    Use this to discover which foods exist / find the exact name before a
    ``macros_lookup`` (which needs a known food)."""
    if not query.strip():
        return "error: a search term is required"
    return _run(["macros", "-q", query])


@mcp.tool()
def macros_add_food(row: str) -> str:
    """Add a new food to the operator's macros database. ``row`` is
    ``<food> <amount><unit>,<protein>,<carbs>,<fat>`` (e.g. "banana 100g,1,23,0.3").

    This WRITES to the database, so later ``macros_lookup``/``macros_search`` calls
    can find it. Use it when a food the user mentions is not yet known."""
    if not row.strip():
        return (
            "error: a food row is required "
            "(<food> <amount><unit>,<protein>,<carbs>,<fat>)"
        )
    return _run(["macros", "-i", row])


def main() -> None:
    """Run the den MCP server over stdio (spawned by a backend on an orchestrator
    turn). Configures its own logging - a separate process from the dashboard."""
    import os

    from .logsetup import configure_logging
    from .mcp_common import _disabled_tools

    configure_logging(os.environ.get("SCUFRIS_LOG_LEVEL", "INFO"))
    removed = apply_disabled_tools(mcp, _disabled_tools())
    if removed:
        logger.info("disabled tools: %s", ", ".join(removed))
    mcp.run()


if __name__ == "__main__":
    main()
