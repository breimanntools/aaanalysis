"""
This is a script for the version-truth guard.

``master`` must never report a version that is already published on PyPI. Both a
development checkout and a released install otherwise answer ``1.0.3`` to
``importlib.metadata.version("aaanalysis")``, which makes two materially
different trees indistinguishable to humans, agents, bug reports, cached
workflows, and scientific reproducibility.

The scheme is a **plain manual version** in ``pyproject.toml`` plus this CI
divergence guard: after each release the maintainer bumps ``[project] version``
to the next unreleased number, and this script asserts that the number is
strictly greater than the latest release published on PyPI. Deriving the version
from git tags (setuptools-scm) was rejected: it produces ``.devN`` / ``+g<sha>``
proliferation.

The published version is resolved from the PyPI JSON API, falling back to the
latest ``vX.Y.Z`` git tag when the network is unavailable (so the guard still
works offline). Requires ``fetch-depth: 0`` in CI for the tag fallback to see
any tags.

This is CI tooling, not library code, so it prints to stdout for the CI log
(the package itself never calls ``print`` -- it uses ``ut.print_out``).

Local use::

    python .github/scripts/check_version_ahead.py             # PyPI, git-tag fallback
    python .github/scripts/check_version_ahead.py --offline    # git tags only

Exit codes: ``0`` the version is ahead (or the check is inconclusive), ``1`` the
version is behind or equal to the latest published release -- bump
``[project] version`` in ``pyproject.toml``.
"""
import sys
import re
import json
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

from packaging.version import Version, InvalidVersion

# The repo root is two levels up from ``.github/scripts/``.
REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
DIST_NAME = "aaanalysis"
PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
PYPI_TIMEOUT_S = 10

# Release tags are ``vX.Y.Z``; the leading ``v`` is optional so the legacy
# ``0.1.1`` tag (predating the convention) still parses instead of being
# silently skipped.
TAG_PATTERN = re.compile(r"^v?(\d+(?:\.\d+)*.*)$")

# ``pyproject.toml`` carries both a ``[project]`` and a ``[tool.poetry]`` block.
# Only ``[project]`` is authoritative, so the fallback parser is section-scoped
# rather than grepping the file for any ``version =`` line.
SECTION_HEADER_PATTERN = re.compile(r"^\s*\[([^\]]+)\]\s*$")
VERSION_ASSIGN_PATTERN = re.compile(r"""^\s*version\s*=\s*["']([^"']+)["']""")


# I Helper Functions
def parse_project_version(path=PYPROJECT_PATH):
    """Return the ``[project] version`` string declared in ``pyproject.toml``.

    Uses ``tomllib`` where available (Python >= 3.11) and falls back to a
    section-scoped scan on the 3.10 floor, so the guard stays stdlib-only across
    the whole supported matrix.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        import tomllib
    except ImportError:                     # Python 3.10 -- no tomllib in stdlib
        return _parse_project_version_fallback(text, path)
    data = tomllib.loads(text)
    try:
        return data["project"]["version"]
    except KeyError:
        raise RuntimeError(f"no [project] version declared in {path}")


def _parse_project_version_fallback(text, path):
    """Return the ``version`` assigned inside the ``[project]`` section of ``text``."""
    in_project = False
    for line in text.splitlines():
        header = SECTION_HEADER_PATTERN.match(line)
        if header:
            in_project = header.group(1).strip() == "project"
            continue
        if in_project:
            match = VERSION_ASSIGN_PATTERN.match(line)
            if match:
                return match.group(1)
    raise RuntimeError(f"no [project] version declared in {path}")


def fetch_latest_pypi_version(name=DIST_NAME, timeout=PYPI_TIMEOUT_S):
    """Return the latest version published on PyPI, or ``None`` if unreachable.

    ``info.version`` is PyPI's own "latest release" pointer, so yanked and
    pre-release uploads do not have to be filtered out by hand.
    """
    url = PYPI_JSON_URL.format(name=name)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        print(f"[version-guard] PyPI unreachable ({e!r}); falling back to git tags.")
        return None
    return payload["info"]["version"]


def latest_git_tag_version(repo_root=REPO_ROOT):
    """Return the highest release version among the repo's git tags, or ``None``.

    Tags are sorted by parsed version rather than by name so ``v1.0.10`` beats
    ``v1.0.9``. Non-release tags that do not parse are skipped.
    """
    try:
        proc = subprocess.run(["git", "tag", "--list"], cwd=str(repo_root),
                              capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"[version-guard] could not list git tags ({e!r}).")
        return None
    versions = []
    for tag in proc.stdout.split():
        match = TAG_PATTERN.match(tag)
        if not match:
            continue
        try:
            versions.append(Version(match.group(1)))
        except InvalidVersion:
            continue
    if not versions:
        return None
    return str(max(versions))


def resolve_published_version(offline=False, name=DIST_NAME, repo_root=REPO_ROOT):
    """Return ``(version_str, source)`` for the latest published release.

    Prefers PyPI (the actual publication record) and falls back to git tags when
    the network is unavailable. ``(None, "unavailable")`` when neither resolves.
    """
    if not offline:
        published = fetch_latest_pypi_version(name=name)
        if published is not None:
            return published, "PyPI"
    published = latest_git_tag_version(repo_root=repo_root)
    if published is not None:
        return published, "git tag"
    return None, "unavailable"


def is_ahead(project_version, published_version):
    """Return True when ``project_version`` is strictly newer than ``published_version``."""
    return Version(project_version) > Version(published_version)


# II Main Functions
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    offline = "--offline" in argv

    # Pass the paths explicitly rather than relying on the def-time default
    # bindings, so the module-level constants are read at call time.
    project_version = parse_project_version(PYPROJECT_PATH)
    published_version, source = resolve_published_version(offline=offline,
                                                          repo_root=REPO_ROOT)

    if published_version is None:
        # Neither PyPI nor a git tag resolved. That is an infrastructure failure,
        # not a version error, so it must not block every merge in the repo --
        # report loudly and pass.
        print(f"[version-guard] INCONCLUSIVE: pyproject is {project_version} but no "
              f"published version could be resolved (PyPI unreachable and no git "
              f"tags). Not failing the build on an infrastructure error.")
        return 0

    print(f"[version-guard] pyproject: {project_version} | "
          f"latest published ({source}): {published_version}")

    if not is_ahead(project_version, published_version):
        print(f"FAIL: pyproject version {project_version} is not ahead of the latest "
              f"published release {published_version}. The development tree would "
              f"report an already-published version, making two different installs "
              f"indistinguishable.\n"
              f"      Fix: bump [project] version in pyproject.toml above "
              f"{published_version}.")
        return 1

    print(f"OK: {project_version} > {published_version} (development tree reports an "
          f"unreleased version).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
