"""Shared pytest fixtures for the aaanalysis test suite."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

import aaanalysis as aa
from aaanalysis.config import _dict_options


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
