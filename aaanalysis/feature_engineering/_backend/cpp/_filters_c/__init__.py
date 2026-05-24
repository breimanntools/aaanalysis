"""
Cython-accelerated kernels for CPP feature-matrix construction.

The compiled extension lives next to ``_inner.pyx``. Wheels published to
PyPI via cibuildwheel ship the precompiled ``.so``; ``pip install
aaanalysis`` users get the fast path with zero ceremony. Source builds
(``pip install -e .``) compile the extension via the setuptools build
backend (see ``setup.py`` at the project root). Import path:

    from aaanalysis.feature_engineering._backend.cpp._filters_c._inner import (
        compute_segment_mean, compute_pattern_n_mean, compute_pattern_c_mean,
    )

When the extension is unavailable (e.g. unsupported platform without a
prebuilt wheel and no local compiler), ``cpp_run._pick_feature_matrix_builder``
silently falls back to the pure-Python ``get_feature_matrix_fast_`` —
bit-exact output, slightly slower.
"""
