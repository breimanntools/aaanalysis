"""Tests for the version-truth guard (``.github/scripts/check_version_ahead.py``).

These exercise the comparison *logic* with fixtures, so no network call is made in
the unit matrix. The KPI pinned here is the issue-#441 acceptance criterion: the
guard fails a build when the pyproject version is <= the latest published PyPI
release, and passes when it is strictly ahead.

The guard is CI tooling rather than library code, so it is loaded from its script
path the same way the perf-regression guard's tests load theirs.
"""
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "check_version_ahead.py"

PYPROJECT_TEMPLATE = """\
[build-system]
requires = ["setuptools"]

[project]
name = "aaanalysis"
version = "{version}"
description = "Python framework for interpretable protein prediction"

[tool.poetry]
name = "aaanalysis"
version = "{poetry_version}"
"""


def _load_module():
    spec = importlib.util.spec_from_file_location("check_version_ahead", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def _write_pyproject(tmp_path, version, poetry_version="0.0.1"):
    path = tmp_path / "pyproject.toml"
    path.write_text(PYPROJECT_TEMPLATE.format(version=version,
                                              poetry_version=poetry_version),
                    encoding="utf-8")
    return path


# I is_ahead() -- the core comparison the guard gates on
class TestIsAhead:
    """Test the version comparison in isolation."""

    def test_equal_version_is_not_ahead(self, mod):
        """KPI: the exact bug -- master declaring the published version."""
        assert mod.is_ahead("1.0.3", "1.0.3") is False

    def test_older_version_is_not_ahead(self, mod):
        assert mod.is_ahead("1.0.2", "1.0.3") is False

    def test_minor_bump_is_ahead(self, mod):
        """KPI: the fix -- 1.1.0 on master against the published 1.0.3."""
        assert mod.is_ahead("1.1.0", "1.0.3") is True

    def test_patch_bump_is_ahead(self, mod):
        assert mod.is_ahead("1.0.4", "1.0.3") is True

    def test_major_bump_is_ahead(self, mod):
        assert mod.is_ahead("2.0.0", "1.0.3") is True

    def test_comparison_is_numeric_not_lexicographic(self, mod):
        """``1.0.10`` beats ``1.0.9`` numerically but loses as a string."""
        assert mod.is_ahead("1.0.10", "1.0.9") is True

    def test_prerelease_is_ahead_of_published_release(self, mod):
        assert mod.is_ahead("1.1.0rc1", "1.0.3") is True

    def test_prerelease_of_published_version_is_not_ahead(self, mod):
        """``1.1.0rc1`` sorts *below* the final ``1.1.0`` under PEP 440."""
        assert mod.is_ahead("1.1.0rc1", "1.1.0") is False


# II parse_project_version() -- reads [project], never [tool.poetry]
class TestParseProjectVersion:
    """Test the pyproject parsing, including the dual-block sharp edge."""

    def test_reads_project_version(self, mod, tmp_path):
        path = _write_pyproject(tmp_path, "1.1.0")
        assert mod.parse_project_version(path) == "1.1.0"

    def test_ignores_tool_poetry_version(self, mod, tmp_path):
        """``pyproject.toml`` carries both blocks; only ``[project]`` is authoritative."""
        path = _write_pyproject(tmp_path, "1.1.0", poetry_version="9.9.9")
        assert mod.parse_project_version(path) == "1.1.0"

    def test_fallback_parser_reads_project_version(self, mod, tmp_path):
        """The 3.10 path (no ``tomllib``) must agree with the ``tomllib`` path."""
        path = _write_pyproject(tmp_path, "1.1.0", poetry_version="9.9.9")
        text = path.read_text(encoding="utf-8")
        assert mod._parse_project_version_fallback(text, path) == "1.1.0"

    def test_missing_project_version_raises(self, mod, tmp_path):
        path = tmp_path / "pyproject.toml"
        path.write_text("[project]\nname = \"aaanalysis\"\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="no \\[project\\] version"):
            mod.parse_project_version(path)

    def test_fallback_missing_project_version_raises(self, mod, tmp_path):
        text = "[tool.poetry]\nversion = \"9.9.9\"\n"
        with pytest.raises(RuntimeError, match="no \\[project\\] version"):
            mod._parse_project_version_fallback(text, tmp_path / "pyproject.toml")


# III latest_git_tag_version() -- the offline fallback + tag-naming tolerance
class TestLatestGitTagVersion:
    """Test the git-tag fallback used when PyPI is unreachable."""

    def test_tag_pattern_accepts_v_prefix(self, mod):
        assert mod.TAG_PATTERN.match("v1.0.3").group(1) == "1.0.3"

    def test_tag_pattern_accepts_legacy_unprefixed_tag(self, mod):
        """The legacy ``0.1.1`` tag predates the ``vX.Y.Z`` convention."""
        assert mod.TAG_PATTERN.match("0.1.1").group(1) == "0.1.1"

    def test_tag_pattern_rejects_non_release_tag(self, mod):
        assert mod.TAG_PATTERN.match("nightly") is None

    def test_picks_highest_version_not_highest_string(self, mod, monkeypatch):
        """Tags sort by parsed version: ``v1.0.10`` > ``v1.0.9``."""
        monkeypatch.setattr(mod.subprocess, "run",
                            lambda *a, **k: _FakeProc("v1.0.9\nv1.0.10\nv1.0.3\n"))
        assert mod.latest_git_tag_version() == "1.0.10"

    def test_mixed_and_unparseable_tags(self, mod, monkeypatch):
        """Non-release tags are skipped; the legacy tag still counts."""
        monkeypatch.setattr(mod.subprocess, "run",
                            lambda *a, **k: _FakeProc("0.1.1\nnightly\nv1.0.3\n"))
        assert mod.latest_git_tag_version() == "1.0.3"

    def test_no_tags_returns_none(self, mod, monkeypatch):
        monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _FakeProc(""))
        assert mod.latest_git_tag_version() is None


class _FakeProc:
    """Minimal stand-in for ``subprocess.run``'s CompletedProcess."""

    def __init__(self, stdout):
        self.stdout = stdout
        self.returncode = 0


# IV main() -- exit codes (the actual CI gate)
class TestMainExitCodes:
    """Test the pass/fail contract CI depends on."""

    def _patch(self, mod, monkeypatch, tmp_path, project_version, published):
        path = _write_pyproject(tmp_path, project_version)
        monkeypatch.setattr(mod, "PYPROJECT_PATH", path)
        monkeypatch.setattr(mod, "resolve_published_version",
                            lambda **kwargs: (published, "PyPI"))

    def test_fails_when_version_equals_latest_release(self, mod, monkeypatch, tmp_path):
        """KPI: the guard fails a build when pyproject version <= latest PyPI release."""
        self._patch(mod, monkeypatch, tmp_path, "1.0.3", "1.0.3")
        assert mod.main([]) == 1

    def test_fails_when_version_behind_latest_release(self, mod, monkeypatch, tmp_path):
        self._patch(mod, monkeypatch, tmp_path, "1.0.2", "1.0.3")
        assert mod.main([]) == 1

    def test_passes_when_version_ahead(self, mod, monkeypatch, tmp_path):
        self._patch(mod, monkeypatch, tmp_path, "1.1.0", "1.0.3")
        assert mod.main([]) == 0

    def test_inconclusive_when_nothing_resolves_does_not_block(self, mod, monkeypatch,
                                                               tmp_path):
        """An infra failure (no PyPI, no tags) must not fail every build."""
        path = _write_pyproject(tmp_path, "1.1.0")
        monkeypatch.setattr(mod, "PYPROJECT_PATH", path)
        monkeypatch.setattr(mod, "resolve_published_version",
                            lambda **kwargs: (None, "unavailable"))
        assert mod.main([]) == 0


# V resolve_published_version() -- PyPI preferred, git tags as fallback
class TestResolvePublishedVersion:
    """Test the source precedence between PyPI and git tags."""

    def test_prefers_pypi_when_reachable(self, mod, monkeypatch):
        monkeypatch.setattr(mod, "fetch_latest_pypi_version", lambda **k: "1.0.3")
        monkeypatch.setattr(mod, "latest_git_tag_version", lambda **k: "0.9.0")
        assert mod.resolve_published_version() == ("1.0.3", "PyPI")

    def test_falls_back_to_git_tag_when_pypi_unreachable(self, mod, monkeypatch):
        monkeypatch.setattr(mod, "fetch_latest_pypi_version", lambda **k: None)
        monkeypatch.setattr(mod, "latest_git_tag_version", lambda **k: "1.0.3")
        assert mod.resolve_published_version() == ("1.0.3", "git tag")

    def test_offline_skips_pypi(self, mod, monkeypatch):
        """``--offline`` must not touch the network at all."""
        def _boom(**kwargs):
            raise AssertionError("PyPI must not be queried when offline=True")
        monkeypatch.setattr(mod, "fetch_latest_pypi_version", _boom)
        monkeypatch.setattr(mod, "latest_git_tag_version", lambda **k: "1.0.3")
        assert mod.resolve_published_version(offline=True) == ("1.0.3", "git tag")

    def test_unavailable_when_neither_resolves(self, mod, monkeypatch):
        monkeypatch.setattr(mod, "fetch_latest_pypi_version", lambda **k: None)
        monkeypatch.setattr(mod, "latest_git_tag_version", lambda **k: None)
        assert mod.resolve_published_version() == (None, "unavailable")


# VI The live repo invariant -- the guard's own subject
class TestRepoVersionIsAhead:
    """Test that this repo's committed version satisfies the guard."""

    def test_committed_version_is_ahead_of_latest_git_tag(self, mod):
        """KPI: master reports an unreleased version, not a published one.

        Compared against git tags rather than PyPI so the unit matrix stays offline.
        Skipped in a tagless (shallow) checkout, where there is nothing to compare.
        """
        published = mod.latest_git_tag_version()
        if published is None:
            pytest.skip("no git tags available (shallow checkout)")
        assert mod.is_ahead(mod.parse_project_version(), published)
