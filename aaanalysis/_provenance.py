"""
This is a script for the frontend of the opt-in provenance record.

# DEV: The record is a plain dict on purpose. It is an extension of the
# reproducibility (random_state) contract, not a result envelope: it never wraps or
# replaces a return value, so every tool keeps returning plain numpy / pandas.
"""
from typing import Optional, Dict, Any, Union
import hashlib
import platform
import subprocess
from pathlib import Path
from importlib.metadata import version as _dist_version, PackageNotFoundError

import numpy as np
import pandas as pd

import aaanalysis.utils as ut


# I Helper Functions
def _get_package_version() -> str:
    """Return ``aaanalysis.__version__``.

    Imported lazily: ``__version__`` is defined in the top-level ``__init__``, which
    imports this module, so a module-level import would be circular. Reading the
    attribute (rather than re-deriving it) keeps the record and ``aa.__version__``
    from ever disagreeing.
    """
    import aaanalysis
    return aaanalysis.__version__


def _get_dependency_versions() -> Dict[str, Optional[str]]:
    """Return ``{name: version}`` for the key dependencies (``None`` when absent)."""
    dict_versions = {}
    for name in ut.LIST_KEY_DEPENDENCIES:
        try:
            dict_versions[name] = _dist_version(name)
        except PackageNotFoundError:
            dict_versions[name] = None
    return dict_versions


def _run_git(package_dir, *args) -> Optional[str]:
    """Return stripped stdout of ``git -C <package_dir> <args>``, or ``None`` on failure."""
    try:
        proc = subprocess.run(["git", "-C", str(package_dir), *args],
                              capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        # git missing, not executable, or timed out -- not a reproducibility error.
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _get_git_commit() -> Optional[str]:
    """Return the git commit of *this package's* checkout, or ``None``.

    Resolvable only for a source / editable install inside a git repository; a regular
    PyPI install yields ``None``, which is the honest answer rather than an error.

    ``git`` searches upward for a repository, so asking from the package directory is
    not enough: a virtualenv commonly lives *inside* the user's own repository (e.g.
    ``<user-repo>/.venv/...``), and an unguarded query there answers with the **user's**
    commit -- misleading provenance, which is worse than none. Being ignored via
    ``.gitignore`` makes no difference. So the repository is accepted only when its root
    is the directory that directly contains this package, which is true for a source
    checkout and false for any install nested under an unrelated repository.
    """
    package_dir = Path(__file__).resolve().parent
    top_level = _run_git(package_dir, "rev-parse", "--show-toplevel")
    if top_level is None:
        return None
    if Path(top_level).resolve() != package_dir.parent:
        return None
    return _run_git(package_dir, "rev-parse", "HEAD")


def _comp_input_hash(data) -> str:
    """Return a stable ``sha256:<hex>`` digest over ``data``.

    Stable across processes and sessions, so two runs of the same input agree.
    Shape / dtype / column names are folded in, so frames that differ only in
    labelling do not collide.
    """
    hasher = hashlib.sha256()
    if isinstance(data, (pd.DataFrame, pd.Series)):
        # hash_pandas_object uses a fixed default key, so it is process-stable
        # (unlike the built-in hash() of a str, which is salted per process).
        hashed = pd.util.hash_pandas_object(data, index=True).values
        hasher.update(np.ascontiguousarray(hashed).tobytes())
        if isinstance(data, pd.DataFrame):
            hasher.update("|".join(map(str, data.columns)).encode("utf-8"))
    else:
        arr = np.asarray(data)
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(str(arr.dtype).encode("utf-8"))
        if arr.dtype == object:
            # Object arrays (e.g. sequence strings) have no stable buffer.
            hasher.update("|".join(map(str, arr.ravel().tolist())).encode("utf-8"))
        else:
            hasher.update(np.ascontiguousarray(arr).tobytes())
    return f"sha256:{hasher.hexdigest()}"


# II Main Functions
def get_provenance(random_state: Optional[int] = None,
                   data: Optional[Union[ut.ArrayLike1D, ut.ArrayLike2D, pd.DataFrame]] = None,
                   ) -> Dict[str, Any]:
    """
    Obtain a provenance record describing how a run can be reproduced.

    Opt-in and side-effect free: nothing in AAanalysis attaches this record to its
    output, and no return type changes. Call it next to a run and keep the record
    with the results.

    The one field external code cannot easily recover is the **effective resolved
    seed**: ``random_state`` is resolved here through the very same check the tools
    use, so the record reports the seed that actually takes effect, including when
    ``options['random_state']`` overrides the value passed in.

    .. versionadded:: 1.1.0

    Parameters
    ----------
    random_state : int, optional
        Seed as it would be passed to a tool (its constructor ``random_state`` or a
        per-call ``seed``). Resolved through the global ``options['random_state']``
        override, so the recorded value is the one that takes effect. ``None`` means
        no seed is in effect (stochastic steps vary between runs).
    data : array-like, pd.DataFrame, or pd.Series, optional
        Input to fingerprint (e.g. ``df_seq``, ``X``, or ``labels``). When given, a
        stable ``sha256`` digest is recorded so a later run can confirm it used the
        same input. When ``None``, ``input_hash`` is ``None``.

    Returns
    -------
    dict_provenance : dict
        JSON-serializable record with the following keys:

        - ``aaanalysis_version``: the installed package version.
        - ``python_version``: the running interpreter version.
        - ``dependencies``: ``{name: version}`` for the dependencies whose version
          can change a computed result (``None`` for any not installed).
        - ``git_commit``: commit of the package checkout, or ``None`` for a regular
          (non-source) install.
        - ``random_state``: the **effective resolved seed**, or ``None``.
        - ``deterministic``: ``True`` when an effective seed is in force, so every
          stochastic step is reproducible; ``False`` when no seed is in effect. A
          tool that uses no randomness is reproducible either way.
        - ``input_hash``: ``sha256:<hex>`` over ``data``, or ``None``.

    Notes
    -----
    The record deliberately carries no timestamp or hostname: every field is
    something that can change a result, which is what makes two records comparable
    for equality as a reproducibility key.

    Examples
    --------
    .. include:: examples/get_provenance.rst
    """
    # Validate (this call is also what resolves the effective seed)
    random_state = ut.check_random_state(random_state=random_state)
    # Build record
    dict_provenance = {"aaanalysis_version": _get_package_version(),
                       "python_version": platform.python_version(),
                       "dependencies": _get_dependency_versions(),
                       "git_commit": _get_git_commit(),
                       "random_state": random_state,
                       "deterministic": random_state is not None,
                       "input_hash": None if data is None else _comp_input_hash(data=data)}
    return dict_provenance
