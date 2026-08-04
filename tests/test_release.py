"""The version, the changelog, and the tag are one fact seen three ways.

These tests are the proof the epic's Done Means asks for: they run against the
REAL `pyproject.toml` and `CHANGELOG.md` in the tree, so a release that would
publish disagreeing versions fails here (and therefore in `nix flake check`
and in CI) before it can reach a tag.

The edge cases run against fixture text instead, because the point of an edge
case is a repository state we do not want to be in.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts import release_tools
from scripts.release_tools import (
    ReleaseError,
    check_agreement,
    cut_changelog,
    find_section,
    is_prerelease,
    main,
    member_pyprojects,
    parse_changelog,
    project_version,
    release_notes,
)
from scufris.version import UNKNOWN_VERSION, __version__

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

PREAMBLE = """\
# Changelog

All notable changes to this project will be documented in this file.

"""


def _doc(*sections: str) -> str:
    return PREAMBLE + "\n\n".join(sections) + "\n"


# --------------------------------------------------------------------------
# The live tree
# --------------------------------------------------------------------------


def test_release_version_sources_agree() -> None:
    """pyproject.toml, CHANGELOG.md's top released section, and the installed
    distribution all name the same version.

    This is the gate the epic names. If you bump `pyproject.toml` without
    cutting the changelog (or the reverse), this fails.
    """
    agreed = check_agreement(
        pyproject_text=PYPROJECT.read_text(encoding="utf-8"),
        changelog_text=CHANGELOG.read_text(encoding="utf-8"),
    )
    assert agreed == project_version(PYPROJECT.read_text(encoding="utf-8"))

    # The installed distribution is the fourth face of the same fact, and it is
    # asserted UNCONDITIONALLY against the version parsed out of pyproject.toml.
    # An earlier draft skipped this when the metadata was missing, which made
    # the check vanish in exactly the situation it exists to catch: the app
    # reporting `0.0.0+unknown` while claiming agreement. If this fails with
    # UNKNOWN_VERSION, the tree under test was never installed - run it through
    # the dev shell or `nix flake check`, which is where the gate lives.
    assert __version__ != UNKNOWN_VERSION, (
        "scufris has no distribution metadata here, so the running version "
        "cannot be compared against pyproject.toml. Run tests from the dev "
        "shell (python -m pytest) or via nix flake check."
    )
    assert __version__ == agreed


def test_live_changelog_has_notes_for_the_current_version() -> None:
    """The version in pyproject.toml has a non-empty, dated changelog section -
    i.e. tagging it today would produce a real release page."""
    version = project_version(PYPROJECT.read_text(encoding="utf-8"))
    notes = release_notes(CHANGELOG.read_text(encoding="utf-8"), version)
    assert notes.strip()


def test_every_workspace_member_shares_the_root_version() -> None:
    """`packages/*/pyproject.toml` names the version the root names.

    The release attaches every wheel `uv build --all-packages` produced to one
    GitHub release, and the root wheel's `Requires-Dist: scufris-core` is
    satisfiable only from that set. A member left at an older version ships a
    set whose parts name different releases.
    """
    members = member_pyprojects(REPO_ROOT)
    assert members, "no packages/*/pyproject.toml found - the workspace is empty"
    agreed = check_agreement(
        pyproject_text=PYPROJECT.read_text(encoding="utf-8"),
        changelog_text=CHANGELOG.read_text(encoding="utf-8"),
        member_texts=members,
    )
    assert all(project_version(text) == agreed for text in members.values())


def test_the_app_pins_hostd_to_one_exact_version() -> None:
    """The app and the helper are two halves of one socket protocol, and since
    the carve they ship from two wheels.

    Nothing else in this repository would notice the pin rotting: with
    `[tool.uv.sources] scufris-hostd = { workspace = true }` uv DROPS the
    version specifier, so `uv lock`, `uv sync` and `nix build` (which resolves
    from the lock) all stay green against a stale `==`. The built wheel carries
    it regardless, so the one consumer it binds - someone installing published
    `scufris` - is the one consumer no gate here can see. Hence a file-based
    check, and hence `scripts/check-release-ready.sh` is not enough.
    """
    import tomllib

    member = REPO_ROOT / "packages" / "hostd" / "pyproject.toml"
    assert member.is_file(), member
    member_version = tomllib.loads(member.read_text(encoding="utf-8"))["project"][
        "version"
    ]

    requirements = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    pins = [
        req for req in requirements if req.split("==")[0].strip() == "scufris-hostd"
    ]
    assert pins == [f"scufris-hostd=={member_version}"], (
        f"the root pyproject must pin scufris-hostd=={member_version}, got {pins}"
    )


def test_a_member_left_behind_is_rejected() -> None:
    with pytest.raises(ReleaseError, match="packages/core/pyproject.toml says 0.0.1"):
        check_agreement(
            pyproject_text=PYPROJECT.read_text(encoding="utf-8"),
            changelog_text=CHANGELOG.read_text(encoding="utf-8"),
            member_texts={
                "packages/core/pyproject.toml": '[project]\nversion = "0.0.1"\n'
            },
        )


def test_a_tag_that_disagrees_is_rejected() -> None:
    with pytest.raises(ReleaseError, match="tag says 9.9.9"):
        check_agreement(
            pyproject_text=PYPROJECT.read_text(encoding="utf-8"),
            changelog_text=CHANGELOG.read_text(encoding="utf-8"),
            tag="v9.9.9",
        )


# --------------------------------------------------------------------------
# Release-notes extraction and its edge cases
# --------------------------------------------------------------------------


def test_release_notes_extraction() -> None:
    doc = _doc(
        "## [Unreleased]",
        "## [1.2.0] - 2026-02-01\n\n### Added\n\n- A second thing.",
        "## [1.1.0] - 2026-01-01\n\n### Fixed\n\n- A first thing.",
    )

    # Exactly that section's body - not the heading, not the next section.
    assert release_notes(doc, "1.2.0") == "### Added\n\n- A second thing."
    assert release_notes(doc, "1.1.0") == "### Fixed\n\n- A first thing."

    # A tag is accepted wherever a version is.
    assert release_notes(doc, "v1.2.0") == release_notes(doc, "1.2.0")

    # No section for the version: refuse, do not publish empty notes.
    with pytest.raises(ReleaseError, match="no section for 3.0.0"):
        release_notes(doc, "3.0.0")

    # The Unreleased section is not a release. Matched on the specific message,
    # not just the word "Unreleased" - which also appears in the "no section"
    # error, so a loose pattern could not fail for the right reason.
    with pytest.raises(ReleaseError, match="resolves to the \\[Unreleased\\]"):
        release_notes(doc, "Unreleased")

    # A section that exists but says nothing is not publishable either.
    empty = _doc("## [Unreleased]", "## [2.0.0] - 2026-03-01\n\n### Added")
    with pytest.raises(ReleaseError, match="is empty"):
        release_notes(empty, "2.0.0")

    # A section with no date was written by hand and half-finished.
    undated = _doc("## [Unreleased]", "## [2.0.0]\n\n- Something.")
    with pytest.raises(ReleaseError, match="no release date"):
        release_notes(undated, "2.0.0")


def test_prerelease_classification_follows_pep_440() -> None:
    """The release page marks a candidate as a pre-release - and only a candidate.

    The release workflow uses this to decide `--prerelease`. A shell regex of
    "anything beyond MAJOR.MINOR.PATCH" was the first attempt and gets `.post1`
    wrong: a post-release comes AFTER the release, not before it, so marking it
    as a candidate would misrepresent it on the release page.
    """
    for final in ["0.1.0", "v0.1.0", "2.0.0", "1.0.0.post1", "v1.2.3.post2"]:
        assert not is_prerelease(final), final
    for candidate in [
        "0.2.0rc1",
        "v0.2.0rc1",
        "1.0.0a1",
        "1.0.0b2",
        "1.0.0.dev4",
        "v0.1.0-rc.1",
        "1.0.0alpha1",
    ]:
        assert is_prerelease(candidate), candidate


def test_release_notes_handles_a_pre_release_suffix() -> None:
    doc = _doc(
        "## [Unreleased]",
        "## [1.0.0rc1] - 2026-02-01\n\n### Added\n\n- A candidate.",
    )
    assert release_notes(doc, "1.0.0rc1") == "### Added\n\n- A candidate."
    assert release_notes(doc, "v1.0.0rc1") == "### Added\n\n- A candidate."


# --------------------------------------------------------------------------
# Cutting the changelog
# --------------------------------------------------------------------------


def test_cut_moves_unreleased_content_into_the_new_version() -> None:
    doc = _doc("## [Unreleased]\n\n### Added\n\n- A new thing.")
    cut = cut_changelog(doc, "1.0.0", "2026-07-29")
    sections = parse_changelog(cut)

    assert [s.version for s in sections] == ["Unreleased", "1.0.0"]
    # Unreleased is emptied and stays open for the next cycle...
    assert sections[0].is_empty
    # ...and its content is now the released version's notes.
    assert release_notes(cut, "1.0.0") == "### Added\n\n- A new thing."
    assert sections[1].date == "2026-07-29"


def test_cut_is_idempotent() -> None:
    doc = _doc("## [Unreleased]\n\n- A thing.")
    once = cut_changelog(doc, "1.0.0", "2026-07-29")
    # A second cut for the same version and date is a no-op - the release
    # pipeline re-runs what the operator already ran by hand. (A different date
    # is a deliberate re-date, not a duplicate; see the redate test.)
    twice = cut_changelog(once, "1.0.0", "2026-07-29")
    assert twice == once
    assert cut_changelog(twice, "1.0.0", "2026-07-29") == once


def test_cut_refuses_an_empty_unreleased_section() -> None:
    doc = _doc("## [Unreleased]", "## [1.0.0] - 2026-07-01\n\n- Old.")
    with pytest.raises(ReleaseError, match="nothing to release"):
        cut_changelog(doc, "1.1.0", "2026-07-29")


def test_cut_maintains_the_link_references() -> None:
    doc = _doc("## [Unreleased]\n\n- Second.", "## [1.0.0] - 2026-07-01\n\n- First.")
    cut = cut_changelog(doc, "1.1.0", "2026-07-29")

    assert (
        "[Unreleased]: https://github.com/alexjercan/scufris/compare/v1.1.0...HEAD"
        in cut
    )
    assert (
        "[1.1.0]: https://github.com/alexjercan/scufris/compare/v1.0.0...v1.1.0" in cut
    )
    # The oldest version has nothing to compare against, so it links to its tag.
    assert "[1.0.0]: https://github.com/alexjercan/scufris/releases/tag/v1.0.0" in cut


def test_cut_handles_an_oddly_spaced_unreleased_heading() -> None:
    """The splice follows the PARSER, not a literal `## [Unreleased]` string.

    A heading with two spaces used to match the parser but not the literal, so
    the cut silently produced a changelog with no released section and no link
    references - and still reported success.
    """
    doc = PREAMBLE + "##  [Unreleased]\n\n- A thing.\n"
    cut = cut_changelog(doc, "1.0.0", "2026-07-29")

    assert [s.version for s in parse_changelog(cut)] == ["Unreleased", "1.0.0"]
    assert release_notes(cut, "1.0.0") == "- A thing."
    assert "[1.0.0]: https://github.com/alexjercan/scufris/releases/tag/v1.0.0" in cut


def test_a_heading_inside_a_code_fence_is_not_a_section() -> None:
    """A changelog entry may quote markdown; those lines are examples.

    The fence and the quoted heading are at COLUMN 0 on purpose. An earlier
    version of this test indented them inside a bullet, where `_SECTION_RE`'s
    `^##` anchor never matched anyway - so the test passed with fence detection
    switched off entirely and proved nothing. Keep them unindented; that is what
    makes this a real proof.
    """
    doc = _doc(
        "## [Unreleased]",
        "## [1.0.0] - 2026-02-01\n\n"
        "### Added\n\n"
        "- A cut writes a heading like this:\n\n"
        "```markdown\n"
        "## [9.9.9] - 2099-01-01\n"
        "```\n\n"
        "- And a second real bullet.",
    )
    assert [s.version for s in parse_changelog(doc)] == ["Unreleased", "1.0.0"]
    notes = release_notes(doc, "1.0.0")
    assert "And a second real bullet." in notes
    with pytest.raises(ReleaseError, match="no section for 9.9.9"):
        release_notes(doc, "9.9.9")


def test_an_unterminated_code_fence_is_an_error_not_a_silent_truncation() -> None:
    """An unclosed fence hides every section below it.

    Left unchecked, a cut would then act on the truncated document: regenerate
    the wrong link references, and fold the previous release's notes into the
    new section - reporting success the whole way. Refuse instead.
    """
    doc = _doc(
        "## [Unreleased]\n\n- A thing:\n\n```py\nprint('never closed')",
        "## [1.0.0] - 2026-01-01\n\n- The previous release.",
    )
    with pytest.raises(ReleaseError, match="unterminated code fence"):
        parse_changelog(doc)
    with pytest.raises(ReleaseError, match="unterminated code fence"):
        cut_changelog(doc, "1.1.0", "2026-07-29")


def test_crlf_input_produces_clean_lf_output() -> None:
    doc = _doc("## [Unreleased]\n\n- A thing.").replace("\n", "\r\n")
    assert "\r" not in release_notes(cut_changelog(doc, "1.0.0", "2026-07-29"), "1.0.0")
    assert "\r" not in cut_changelog(doc, "1.0.0", "2026-07-29")


def test_an_already_cut_section_can_be_redated_only_on_request() -> None:
    """A release that slips a day must not stay stamped with its draft date -
    but re-dating is an explicit act, never a side effect of re-running."""
    doc = _doc("## [Unreleased]\n\n- A thing.")
    cut = cut_changelog(doc, "1.0.0", "2026-07-29")

    # Default: a later date changes nothing at all.
    assert cut_changelog(cut, "1.0.0", "2026-08-02") == cut

    moved = cut_changelog(cut, "1.0.0", "2026-08-02", redate=True)
    section = find_section(parse_changelog(moved), "1.0.0")
    assert section is not None and section.date == "2026-08-02"
    assert release_notes(moved, "1.0.0") == "- A thing."


def test_a_dateless_rerun_of_the_cut_command_never_moves_the_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`cut-changelog.sh X.Y.Z` twice, on different days, is byte-identical.

    This is the invariant the Definition of Done names ("the changelog cut is
    scripted and idempotent"), asserted over `main()` rather than over
    `cut_changelog`, because it was `main()`'s date defaulting that broke it:
    `date = args.date or today()` handed a NEW date to a function that would
    re-date on any difference, so a dateless re-run silently moved a published
    version's release date and still exited 0.
    """
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "scufris"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(_doc("## [Unreleased]\n\n- A thing."), encoding="utf-8")

    monkeypatch.setattr(release_tools, "today", lambda: "2026-07-29")
    assert main(["--root", str(tmp_path), "cut", "1.0.0"]) == 0
    after_first = changelog.read_text(encoding="utf-8")
    assert "## [1.0.0] - 2026-07-29" in after_first

    # Two weeks later, the release pipeline re-runs the same command.
    monkeypatch.setattr(release_tools, "today", lambda: "2026-08-15")
    assert main(["--root", str(tmp_path), "cut", "1.0.0"]) == 0
    assert changelog.read_text(encoding="utf-8") == after_first

    # An explicit --date is the operator saying so, and DOES move it.
    assert main(["--root", str(tmp_path), "cut", "1.0.0", "--date", "2026-08-15"]) == 0
    assert "## [1.0.0] - 2026-08-15" in changelog.read_text(encoding="utf-8")

    # ...but only a real date. --date is the one gesture that authorizes
    # overwriting a published version's date, so a typo must not re-date the
    # section AND write nonsense into it in one step.
    before = changelog.read_text(encoding="utf-8")
    assert main(["--root", str(tmp_path), "cut", "1.0.0", "--date", "bananas"]) == 1
    assert changelog.read_text(encoding="utf-8") == before


def test_cut_replaces_a_stale_v_prefixed_link_reference() -> None:
    doc = (
        _doc("## [Unreleased]\n\n- A thing.")
        + "\n[v1.0.0]: https://example.invalid/stale\n"
    )
    cut = cut_changelog(doc, "1.0.0", "2026-07-29")
    assert "example.invalid" not in cut
    assert cut.count("1.0.0]: ") == 1


def test_cut_output_still_agrees_with_a_bumped_pyproject() -> None:
    """The cut is what MAKES the sources agree - the whole point of scripting it."""
    doc = _doc("## [Unreleased]\n\n- A thing.")
    pyproject = '[project]\nname = "scufris"\nversion = "1.4.0"\n'

    with pytest.raises(ReleaseError, match="no released section"):
        check_agreement(pyproject_text=pyproject, changelog_text=doc)

    cut = cut_changelog(doc, "1.4.0", "2026-07-29")
    assert (
        check_agreement(pyproject_text=pyproject, changelog_text=cut, tag="v1.4.0")
        == "1.4.0"
    )


# --------------------------------------------------------------------------
# The running application
# --------------------------------------------------------------------------


def test_app_reports_its_version() -> None:
    """The version is reported by the app the way an operator actually sees it:
    over the API, and from the command line."""
    from scufris.app import SCUFRIS_VERSION, create_app
    from scufris.config import Settings

    # Compared against pyproject.toml, NOT against scufris_version() - asserting
    # the app agrees with the same call it uses would pass just as happily while
    # everything reported "0.0.0+unknown".
    expected = project_version(PYPROJECT.read_text(encoding="utf-8"))

    assert SCUFRIS_VERSION == expected

    app = create_app(settings=Settings(_env_file=None))  # type: ignore[call-arg]
    assert app.version == expected

    with TestClient(app) as client:
        response = client.get("/api/agents/orchestrator/health")
        assert response.status_code == 200
        assert response.json()["scufris_version"] == expected


def test_cli_version_flag_prints_the_version() -> None:
    """`scufris --version` is what the release pipeline smoke-tests against a
    freshly built wheel, so it must work as a plain subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "scufris", "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    expected = project_version(PYPROJECT.read_text(encoding="utf-8"))
    assert result.stdout.strip() == f"scufris {expected}"
