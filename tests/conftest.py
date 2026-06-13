"""Shared pytest fixtures for the aaanalysis test suite."""
import matplotlib as mpl
mpl.use("Agg")  # headless, deterministic backend -> lower render-time variance
import matplotlib.pyplot as plt
import pytest

import aaanalysis as aa
from aaanalysis.config import _dict_options


# Live-endpoint ('network'-marked) tests are skipped by default so they never run
# in the blocking matrix (a per-job 'pytest -m "not regression"' would otherwise
# hit AlphaFold/UniProt live on every push). Run them with '--run-network' (the
# nightly does) to catch upstream API/version breakage. A code-side regression is
# still caught networklessly by the mocked unit tests.
def pytest_addoption(parser):
    parser.addoption("--run-network", action="store_true", default=False,
                     help="run 'network'-marked tests against live endpoints")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="live endpoint; pass --run-network to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture(scope="session", autouse=True)
def _warm_matplotlib():
    """Prime matplotlib's font cache + first-figure cost ONCE per session.

    The plotting Hypothesis tests fail intermittently with
    ``FlakyFailure: DeadlineExceeded`` on the **first (cold) example** (#83):
    building the FreeType font cache and initialising the first figure in a
    fresh process can take several seconds, and Hypothesis charges that one-time
    cost to whichever example happens to render first. Steady-state render is
    ~0.1-1.7s (well within the per-example deadlines), so the deadlines stay
    meaningful once the cold cost is paid up front here, outside any timed test.
    """
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "warmup AAanalysis 0123")  # force text -> build font cache
    fig.canvas.draw()
    plt.close(fig)
    yield


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Reset global state to package defaults around every test.

    Two leaks this guards against:

    * **matplotlib rcParams** — a single ``aa.plot_settings(font_scale=0)``
      call (which the hypothesis cases in ``test_plot_settings.py`` can draw,
      ``min_value=0``) leaves the font rcParams at 0.0, and every later test
      that calls ``plt.tight_layout`` crashes inside FreeType with
      ``error code 0x97`` (``FT_Err_Invalid_Pixel_Size``).
    * **aa.options** — many test modules set ``aa.options["verbose"] = False``
      at *module level*, which runs at import/collection time. Because options
      override per-call args (the ``"off"`` contract in ``config.py``), a
      leaked ``verbose=False`` makes e.g. ``aa.CPP(verbose="invalid")`` stop
      raising, so ``test_invalid_verbose`` fails whenever its sibling modules
      are collected with it. We therefore reset to the package **defaults**
      (``_dict_options``), not to a snapshot of the already-polluted state.

    Resetting before *and* after `yield` keeps tests independent of collection
    order regardless of what import-time module code mutated.
    """
    saved_rc = mpl.rcParams.copy()
    # Bypass `__setitem__` validators: we write defaults directly, and a
    # known-bug validator (`_check_option` for `ext_len` lacks `just_int=True`)
    # rejects the default `0` if routed through the public API.
    aa.options._settings.update(_dict_options)
    try:
        yield
    finally:
        plt.close("all")
        mpl.rcParams.update(saved_rc)
        aa.options._settings.update(_dict_options)


@pytest.fixture(scope="session", autouse=True)
def _prime_cpp_fallback_notice():
    """Mark the one-time CPP Python-kernel fallback notice as already shown.

    On an install without the compiled Cython extension,
    ``_pick_feature_matrix_builder`` emits a one-time ``UserWarning`` on its
    first call (issue #74). Several CPP tests run ``cpp.run(...)`` inside
    ``warnings.simplefilter("error", UserWarning)`` to assert their *own* code
    path is warning-free; whichever of them happened to trigger the very first
    feature-matrix build would otherwise turn that orthogonal install-state
    notice into an error (flaky, ordering-dependent, non-Cython only). Priming
    the guard once per session keeps those assertions about the intended
    warnings only. The dedicated test in ``test_cpp_run_fallback_notice.py``
    flips the guard back off to exercise the firing path explicitly.
    """
    import aaanalysis.feature_engineering._backend.cpp_run as _cpp_run
    _cpp_run._PYTHON_FALLBACK_NOTIFIED = True
    yield
