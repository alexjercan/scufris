"""Tests for the `den` MCP server (scufris.den_mcp_server): the-den journal +
macros life tools.

Two layers, as before: (1) gating + argv-construction tests that need NO
`today`/`macros` binary (they short-circuit on a bad den, or capture the argv by
stubbing ``_run``), so they pin the exact CLI contract deterministically and stay
green in the pure `nix flake check` sandbox; (2) real end-to-end tests that drive
the ACTUAL CLIs against a temp den/DB, skipped where the CLI is absent. The den
is injected as SCUFRIS_DEN_PATH, exactly as agent.scufris_mcp_servers does. Tools
are called directly (FastMCP's decorator returns the original function).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scufris.den_mcp_server import (
    journal_add_macros,
    journal_add_note,
    journal_add_task,
    journal_complete_task,
    journal_log_weight,
    journal_notes,
    journal_remove_task,
    journal_show,
    journal_toggle_habit,
    macros_add_food,
    macros_lookup,
    macros_search,
    mcp,
)


async def test_den_server_exposes_only_life_tools() -> None:
    names = {tool.name for tool in await mcp.list_tools()}
    assert names == {
        "journal_show",
        "journal_notes",
        "journal_add_task",
        "journal_complete_task",
        "journal_remove_task",
        "journal_toggle_habit",
        "journal_log_weight",
        "journal_add_macros",
        "journal_add_note",
        "macros_lookup",
        "macros_search",
        "macros_add_food",
    }
    assert all(tool.description for tool in await mcp.list_tools())


_HAS_TODAY = shutil.which("today") is not None
requires_today = pytest.mark.skipif(
    not _HAS_TODAY, reason="the `today` CLI is not on PATH (journal end-to-end tests)"
)

# The daily template `today` renders a new entry from; carrying the real den's
# Habits/Macros/Notes sections so a fresh temp den has habits to toggle.
_DAILY_TEMPLATE = """# {{title}}

### 🌱 Habits

- [ ] 📕 Learn
- [ ] 💪 Gym
- [ ] 🥕 Track Macros

### 🍽️ Macros

what,protein,carbs,fat

### 📝 Notes
"""


@pytest.fixture
def den(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp den (with the daily template) wired as SCUFRIS_DEN_PATH."""
    root = tmp_path / "the-den"
    (root / "Templates").mkdir(parents=True)
    (root / "Templates" / "daily.md").write_text(_DAILY_TEMPLATE)
    monkeypatch.setenv("SCUFRIS_DEN_PATH", str(root))
    return root


# --- gating: no den configured / den missing (no `today` needed) --------------


def _record_run(sink: list[list[str]], ret: str):
    """A `_run` stand-in that records the argv it was handed and returns ``ret`` -
    so a test can assert the tool did (or did not) shell out, and with what."""

    def _fake(args: list[str], **_kw: object) -> str:
        sink.append(args)
        return ret

    return _fake


def test_journal_unconfigured_reports_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    """With SCUFRIS_DEN_PATH unset the tools are inert: a clear message and no
    shell-out (so scufris is safe on a box without the-den)."""
    monkeypatch.delenv("SCUFRIS_DEN_PATH", raising=False)
    called: list[list[str]] = []
    monkeypatch.setattr("scufris.den_mcp_server._run", _record_run(called, ""))
    out = journal_show()
    assert out.startswith("error:") and "not configured" in out
    assert called == []  # never shelled out


def test_journal_missing_den_reports_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured-but-absent den short-circuits with a clean error instead of
    letting the raw CLI raise a traceback."""
    monkeypatch.setenv("SCUFRIS_DEN_PATH", "/no/such/den/xyz")
    called: list[list[str]] = []
    monkeypatch.setattr("scufris.den_mcp_server._run", _record_run(called, ""))
    out = journal_add_task("buy milk")
    assert out.startswith("error:") and "does not exist" in out
    assert called == []


def test_journal_expands_tilde_in_den(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `~`-prefixed SCUFRIS_DEN_PATH is expanded at use time (repo convention:
    pydantic stores env Paths verbatim, consumers expanduser), so the
    `~/personal/the-den` form documented in .env.example actually works - both the
    dir check and the `--den` arg see the resolved absolute path, never a `~`."""
    (tmp_path / "den").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SCUFRIS_DEN_PATH", "~/den")
    seen: list[list[str]] = []
    monkeypatch.setattr("scufris.den_mcp_server._run", _record_run(seen, "{}"))
    journal_show()
    assert seen and seen[0][2] == str(tmp_path / "den")  # --den arg, expanded
    assert "~" not in seen[0][2]


# --- argv construction (no `today` needed: `_run` is stubbed) -----------------


@pytest.fixture
def capture_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Point SCUFRIS_DEN_PATH at a real (empty) dir so the den check passes, then
    capture the argv each tool hands to `_run` without invoking `today`."""
    monkeypatch.setenv("SCUFRIS_DEN_PATH", str(tmp_path))
    seen: list[list[str]] = []
    monkeypatch.setattr("scufris.den_mcp_server._run", _record_run(seen, "{}"))
    return seen


def test_journal_argv_contract(capture_run: list[list[str]]) -> None:
    """Every tool builds the exact `today --den <den> ...` argv, with global flags
    (--den, -N) BEFORE the subcommand as the CLI requires."""
    journal_show()
    journal_show(offset=-1)
    journal_notes()
    journal_notes(tag="mood")
    journal_add_task("buy milk")
    journal_add_task("call bob", tomorrow=True)
    journal_complete_task(2)
    journal_remove_task(1)
    journal_remove_task(3, tomorrow=True)
    journal_toggle_habit("Gym")
    journal_log_weight("80kg")
    journal_add_macros("eggs,20,2,15")
    journal_add_note("felt great")
    journal_add_note("tagged", tag="mood")
    # Strip the leading `today --den <den>` prefix each call shares.
    tails = [args[3:] for args in capture_run]
    assert all(args[0] == "today" and args[1] == "--den" for args in capture_run)
    assert tails == [
        ["-N", "0", "show", "--json"],
        ["-N", "-1", "show", "--json"],
        ["note", "list", "--json"],
        ["note", "list", "--tag", "mood", "--json"],
        ["task", "add", "buy milk", "--json"],
        ["task", "add", "call bob", "--tomorrow", "--json"],
        ["task", "done", "2", "--json"],
        ["task", "rm", "1", "--json"],
        ["task", "rm", "3", "--tomorrow", "--json"],
        ["habit", "toggle", "Gym", "--json"],
        ["weight", "80kg"],
        ["macros", "add", "eggs,20,2,15", "--json"],
        ["note", "add", "felt great", "--json"],
        ["note", "add", "tagged", "--tag", "mood", "--json"],
    ]


def test_journal_input_guards(capture_run: list[list[str]]) -> None:
    """Empty required text is rejected before shelling out."""
    assert journal_add_task("  ").startswith("error:")
    assert journal_toggle_habit("").startswith("error:")
    assert journal_log_weight("").startswith("error:")
    assert journal_add_macros("").startswith("error:")
    assert journal_add_note("").startswith("error:")
    assert capture_run == []  # none reached `_run`


# --- real `today` CLI against a temp den --------------------------------------


@requires_today
def test_journal_show_reads_the_day(den: Path) -> None:
    data = json.loads(journal_show())
    assert data["date"]  # the CLI stamped a dated entry
    names = {h["name"] for h in data["habits"]}
    assert any("Gym" in n for n in names)
    assert data["tasks"] == [] and data["tomorrow"] == []


@requires_today
def test_journal_task_lifecycle(den: Path) -> None:
    added = json.loads(journal_add_task("buy milk"))
    assert added == [{"index": 1, "text": "buy milk", "done": False}]
    done = json.loads(journal_complete_task(1))
    assert done[0]["done"] is True
    # The mutation is durable: a fresh show reflects it.
    assert json.loads(journal_show())["tasks"][0]["done"] is True
    assert json.loads(journal_remove_task(1)) == []


@requires_today
def test_journal_tomorrow_task(den: Path) -> None:
    added = json.loads(journal_add_task("call bob", tomorrow=True))
    assert added == [{"index": 1, "text": "call bob"}]
    assert json.loads(journal_show())["tomorrow"][0]["text"] == "call bob"
    assert json.loads(journal_remove_task(1, tomorrow=True)) == []


@requires_today
def test_journal_toggle_habit(den: Path) -> None:
    habits = json.loads(journal_toggle_habit("Gym"))
    gym = next(h for h in habits if "Gym" in h["name"])
    assert gym["done"] is True


@requires_today
def test_journal_log_weight_then_show(den: Path) -> None:
    out = journal_log_weight("80kg")
    assert "80" in out
    assert json.loads(journal_show())["weight"] == 80.0


@requires_today
def test_journal_add_macros(den: Path) -> None:
    agg = json.loads(journal_add_macros("eggs,20,2,15"))
    assert agg["protein"] == 20.0 and agg["carbs"] == 2.0 and agg["fat"] == 15.0


@requires_today
def test_journal_notes_add_and_filter(den: Path) -> None:
    journal_add_note("felt great", tag="mood")
    journal_add_note("bought a book")
    assert len(json.loads(journal_notes())) == 2
    mood = json.loads(journal_notes(tag="mood"))
    assert mood == [{"text": "felt great", "tag": "mood"}]


@requires_today
def test_journal_bad_index_is_clean_error(den: Path) -> None:
    """A bad task index surfaces the CLI's one-line stderr, not a traceback."""
    out = journal_complete_task(99)
    assert "no today task" in out.lower() or "error" in out.lower()
    assert "Traceback" not in out


# --- macros food-lookup tools -------------------------------------------------


_HAS_MACROS = shutil.which("macros") is not None
requires_macros = pytest.mark.skipif(
    not _HAS_MACROS, reason="the `macros` CLI is not on PATH (macros end-to-end tests)"
)


def test_macros_argv_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each tool builds the exact `macros` argv: a bare query for lookup, `-q` for
    search, `-i` for the write."""
    seen: list[list[str]] = []
    monkeypatch.setattr("scufris.den_mcp_server._run", _record_run(seen, "ok"))
    macros_lookup("egg 2p")
    macros_search("chick")
    macros_add_food("banana 100g,1,23,0.3")
    assert seen == [
        ["macros", "egg 2p"],
        ["macros", "-q", "chick"],
        ["macros", "-i", "banana 100g,1,23,0.3"],
    ]


def test_macros_input_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty input is rejected before shelling out."""
    called: list[list[str]] = []
    monkeypatch.setattr("scufris.den_mcp_server._run", _record_run(called, "ok"))
    assert macros_lookup("  ").startswith("error:")
    assert macros_search("").startswith("error:")
    assert macros_add_food("   ").startswith("error:")
    assert called == []


@pytest.fixture
def macros_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp HOME with a seeded macros DB so the real-CLI tests are hermetic (they
    do not read/write the operator's live ~/.local/share/nvim/macros.csv). The
    `macros` CLI resolves its DB from $HOME, and `_run` runs it with the inherited
    env, so monkeypatching HOME redirects it to this temp DB."""
    db = tmp_path / ".local" / "share" / "nvim" / "macros.csv"
    db.parent.mkdir(parents=True)
    db.write_text("egg 1pc,6,0,5\nbanana 100g,1,23,0.3\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    return db


@requires_macros
def test_macros_lookup_returns_csv_row(macros_home: Path) -> None:
    """A real lookup returns the `<food> <amount><unit>,<protein>,<carbs>,<fat>`
    row - the exact shape journal_add_macros consumes - scaled to the amount."""
    out = macros_lookup("egg 2p").strip()
    assert out == "egg 2pc,12,0,10"  # 1pc (6,0,5) scaled to 2pc


@requires_macros
def test_macros_search_lists_matches(macros_home: Path) -> None:
    matches = macros_search("ban")
    assert "banana" in matches.lower()


@requires_macros
def test_macros_add_food_then_lookup_finds_it(macros_home: Path) -> None:
    """add_food writes to the DB (a real insert into the temp copy), so a later
    lookup resolves the new food."""
    added = macros_add_food("oats 40g,5,27,3")
    assert "oats" in added.lower()
    assert macros_lookup("oats 40g").strip() == "oats 40g,5,27,3"


@requires_macros
def test_macros_lookup_unknown_food_is_clean_error(macros_home: Path) -> None:
    """An unknown food surfaces the CLI's message, not a traceback."""
    out = macros_lookup("zzznotarealfood 1p")
    assert "unknown food" in out.lower() or "error" in out.lower()
    assert "Traceback" not in out
