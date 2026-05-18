"""Shared pytest fixtures for the aaanalysis test suite."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

import aaanalysis as aa


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Snapshot and restore global state around every test.

    Without this, a single ``aa.plot_settings(font_scale=0)`` call (which the
    hypothesis cases in ``test_plot_settings.py`` can draw because the
    strategy uses ``min_value=0``) leaves matplotlib's font rcParams at 0.0,
    and every later test that calls ``plt.tight_layout`` crashes inside
    FreeType with ``error code 0x97`` (``FT_Err_Invalid_Pixel_Size``).

    Restoring ``mpl.rcParams`` + ``aa.options`` and closing all figures
    between tests keeps tests independent of collection order.
    """
    saved_rc = mpl.rcParams.copy()
    saved_options = aa.options._settings.copy()
    try:
        yield
    finally:
        plt.close("all")
        mpl.rcParams.update(saved_rc)
        # Bypass `__setitem__` validators on restore: the values we're putting
        # back were valid when originally set, and a known-bug validator
        # (`_check_option` for `ext_len` lacks `just_int=True`) rejects the
        # default `0` if routed through the public API.
        aa.options._settings.update(saved_options)
